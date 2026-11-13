import argparse
import json
import random

import wandb
wandb.login()

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from tensorboard_logger import configure

import os
import time
import copy

import gymnasium as gym
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


# Check Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# device = torch.device("cpu")

dtype = torch.float32


##### Online Natural Policy Gradient optimizer ##################################
class OnlineNPG(torch.optim.Optimizer):
# class OnlineNPG:
    """
    Online Natural Policy Gradient (Definition 1).
 
    In iteration i, with N state-action transitions (s_ij, a_ij) and advantage
    estimates A_hat_ij collected under pi_{theta_{i-1}}:
 
        g_hat_i = 1/(N(1-gamma)) * sum_j grad log pi_{theta_{i-1}}(a_ij|s_ij) * A_hat_ij     (12a)
        F_hat_i = 1/N * sum_j grad log pi_{theta_{i-1}}(a_ij|s_ij)
                              grad log pi_{theta_{i-1}}(a_ij|s_ij)^T                          (12b)
        F_bar_i = (1 - beta_i) * F_bar_{i-1} + beta_i * F_hat_i                               (13a)
        theta_i = theta_{i-1} + eta_i * F_bar_i^{-1} g_hat_i                                  (13b)
 
    Everything is computed exactly: one autograd pass per transition for the
    per-sample score vectors, the full d x d Fisher matrix, and an exact linear
    solve for F_bar_i^{-1} g_hat_i.
    """
 
    def __init__(self, params, lr, beta, gamma, init_fisher=1.0, fisher_dtype=torch.float64):
        """
        params       : iterable of policy parameters; their flattened concatenation is theta.
        lr           : learning rate eta_i, a float or a callable i -> float.
        beta         : Fisher moving-average weight beta_i in (0, 1],
                       a float or a callable i -> float.
        gamma        : discount factor in [0, 1).
        init_fisher  : initial Fisher F_bar_0, a scalar c (F_bar_0 = c * I) or a (d, d) tensor.
        fisher_dtype : dtype used to store F_bar and perform the solve (numerical precision only).
        """
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be in [0, 1).")

        params = [p for p in params if p.requires_grad]
        super().__init__(params, dict(lr=lr, beta=beta, gamma=gamma))

        self.gamma = gamma
        self.lr = lr
        self.beta = beta

        self.params = params
        theta = parameters_to_vector(self.params)
        self.d = theta.numel()
        self.dtype, self.device = theta.dtype, theta.device
        # MPS has no float64 / linalg.solve -> keep Fisher on CPU there
        self.fisher_device = torch.device("cpu") if self.device.type == "mps" else self.device
        self.fisher_dtype = fisher_dtype

        if torch.is_tensor(init_fisher):
            F0 = init_fisher
        else:
            F0 = float(init_fisher) * torch.eye(self.d)
        F0 = torch.as_tensor(F0, dtype=self.fisher_dtype, device=self.fisher_device)
        if F0.shape != (self.d, self.d):
            raise ValueError(f"F0 must have shape ({self.d}, {self.d}).")
        self.F_bar = F0.detach().clone()

        self.i = 0  # iteration counter

    @staticmethod
    def _schedule(value, i):
        return value(i) if callable(value) else value

    def _score_vectors(self, log_probs):
        """Rows are grad_theta log pi_{theta_{i-1}}(a_ij | s_ij), shape (N, d)."""
        log_probs = log_probs.reshape(-1)
        N = log_probs.shape[0]
        try:
            # N backward passes (rows of the identity), vectorized
            eye = torch.eye(N, dtype=log_probs.dtype, device=log_probs.device)
            grads = torch.autograd.grad(log_probs, self.params, grad_outputs=eye,
                                        is_grads_batched=True, allow_unused=True)
            return torch.cat([
                (torch.zeros(N, p.numel(), dtype=self.dtype, device=self.device) if g is None else g.reshape(N, -1))
                for g, p in zip(grads, self.params)
            ], dim=1)
        except RuntimeError:
            # Same result, one autograd pass per transition
            scores = torch.empty(N, self.d, dtype=self.dtype, device=self.device)
            for j in range(N):
                grads = torch.autograd.grad(
                    log_probs[j], self.params,
                    retain_graph=(j < N - 1), allow_unused=True,
                )
                scores[j] = torch.cat([
                    (torch.zeros_like(p) if g is None else g).reshape(-1)
                    for g, p in zip(grads, self.params)
                ])
            return scores

    def update(self, advantages, log_probs):
        """
        Performs one Online NPG iteration i using the mini-batch of N transitions.
        advantages : (N,) advantage estimates A_hat_ij.
        log_probs  : (N,) log pi_{theta_{i-1}}(a_ij | s_ij), still attached to the graph.
        Returns theta_i as a flat tensor.
        """
        self.i += 1
        eta_i = self._schedule(self.lr, self.i)
        beta_i = self._schedule(self.beta, self.i)
        if not 0.0 < beta_i <= 1.0:
            raise ValueError("beta_i must be in (0, 1].")

        S = self._score_vectors(log_probs).detach()  # (N, d)
        N = S.shape[0]
        A_hat = torch.as_tensor(advantages, dtype=self.dtype, device=self.device).detach().reshape(-1)
        if A_hat.shape[0] != N:
            raise ValueError("Need one advantage estimate per transition.")

        S = S.to(device=self.fisher_device, dtype=self.fisher_dtype)
        A_hat = A_hat.to(device=self.fisher_device, dtype=self.fisher_dtype)

        g_hat = S.T @ A_hat / (N * (1.0 - self.gamma))  # (12a)
        # (12b) F_hat = S^T S / N  and  (13a) F_bar = (1 - beta) F_bar + beta F_hat,
        # fused in place to avoid allocating a second d x d matrix
        self.F_bar.mul_(1.0 - beta_i).addmm_(S.T, S, alpha=beta_i / N)

        with torch.no_grad():
            theta = parameters_to_vector(self.params)
            # F_bar_i^{-1} g_hat_i : exact solve (F_bar_i is SPD -> Cholesky; LU fallback)
            L, info = torch.linalg.cholesky_ex(self.F_bar)
            if info.item() == 0:
                nat_grad = torch.cholesky_solve(g_hat.unsqueeze(1), L).squeeze(1)
            else:
                nat_grad = torch.linalg.solve(self.F_bar, g_hat)
            theta = theta + eta_i * nat_grad.to(device=self.device, dtype=self.dtype)  # (13b)
            vector_to_parameters(theta, self.params)

        return theta.clone()

    def step(self, closure=None):
        raise NotImplementedError("Use OnlineNPG.update(advantages, log_probs).")

optimizers = {
    'online-npg': OnlineNPG,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Experiment name', default='AUTO')
    # parser.add_argument('--model', type=str, help='Model name: resnet18', default='resnet18_cifar')
    parser.add_argument('--optim', type=str, help='Optimizer name: online-npg',
                        choices=list(optimizers.keys()), default='online-npg')
    parser.add_argument('--env_id', type=str, help='Environment to run the experiment on', default='Reacher-v4')  # default='MountainCarContinuous-v0'
    parser.add_argument('--total-timesteps', type=int, help='total timesteps of the experiments', default=5000000) # original default: 1000000
    parser.add_argument('--num-envs', type=int, help='the number of parallel game environments', default=4) # original default: 1
    parser.add_argument('--num-steps', type=int, help='the number of steps to run in each environment per policy rollout', default=128)  #32    # default=2048
    parser.add_argument('--gamma', type=float, help='the discount factor gamma', default=0.99)
    parser.add_argument('--gae-lambda', type=float, help='the lambda for the general advantage estimation', default=0.95)
    # parser.add_argument('--batch-size', type=int, help='Batch size', default=32)
    parser.add_argument('--num-minibatches', type=int, help='Number of Mini Batch', default=1)     # original default: 32
    parser.add_argument('--update-epochs', type=int, help='the K epochs to update the policy', default=1) # original default: 10
    parser.add_argument('--norm-adv', type=bool, help='Toggles advantages normalization', default=True) # default=True
    parser.add_argument('--ent-coef', type=float, help='coefficient of the entropy', default=0.0)
    parser.add_argument('--vf-coef', type=float, help='coefficient of the value function', default=0.5)
    parser.add_argument('--num-gpus', type=int, help='Number of gpus', default=1)
    parser.add_argument('--num-epochs', type=int, help='Number of epochs', default=1) # original default: 100
    parser.add_argument('--torch-deterministic', type=bool, help='no idea!', default=True)
    parser.add_argument('--capture-video', type=bool, help='whether to capture videos of the agent performances (check out `videos` folder)', default=False)
    parser.add_argument('--saving_freq', type=int, help='Frequency of model checkpoints', default=100)
    parser.add_argument('--resume-network', type=str, help='Path to model checkpoint to resume', default=None)
    parser.add_argument('--resume-epoch', type=int, help='Starting with epoch', default=None)
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--data-dir', type=str, default='./data', help='Dataset location')
    parser.add_argument('--warmup-epochs', type=int, help='Warmup epochs, set to 0 to disable', default=0)
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'step'], help='lr scheduler type') # original default: step
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD's momentum")
    parser.add_argument('--step-gamma', type=float, default=0.2, help="step scheduler's gamma (lr*=gamma)")
    parser.add_argument('--seed', type=int, help='random seed', default=42)

    ## Value optimization
    parser.add_argument('--v-lr', type=float, default=1e-4, help='learning rate for the value optimizer')  # default=5e-4
    parser.add_argument('--v-epochs', type=int, help='the epochs to update the value network', default=5)
    parser.add_argument('--v-minibatches', type=int, help='Number of Mini Batch for value network', default=4)
    parser.add_argument('--max-grad-norm', type=float, help='Max gradient norm for clipping', default=5.0)

    ## Importance sampling
    parser.add_argument('--do-importance-sampling', type=bool, default=False)

    ## Online NPG specific args **************************************************************************
    parser.add_argument('--eta', type=float, default=5e-4, help='Online NPG learning rate eta_i')                   # default=1e-3
    parser.add_argument('--beta', type=float, default=0.001, help='Fisher moving average weight beta_i in (0, 1]')   # default=0.1 (or 0.01 or 0.001)
    parser.add_argument('--init-fisher', type=float, default=1.0, help='Initial Fisher F_bar_0 = init_fisher * I')  # default=1.0
    parser.add_argument('--fisher-dtype', type=str, default='float64', choices=['float32', 'float64'], help='dtype of F_bar and the linear solve (numerical precision only)')
    return parser.parse_args()


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env) 
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


##### continuous action agent ##################################
class Agent_C(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), probs


##### discrete action agent ##################################
class Agent_D(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), probs


def main():
    args = parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size


    if args.exp == 'AUTO':
        args.exp = f'{args.optim} eta:{args.eta} beta:{args.beta} F0:{args.init_fisher}'                ################################ Here is the Legend Explanation ################################
    
    wandb.init(
        project = f'{args.env_id} (2)', # project name 
        entity="hossein_abdi-the-university-of-manchester",
        name=args.exp,
        config=vars(args),                   # command line arguments
        monitor_gym=True,
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    configure('runs/{}'.format(args.exp))

    with open(f'runs/{args.exp}/args.json', 'wt') as f:
        json.dump(vars(args), f, indent=2)

    writer = SummaryWriter(f"runs/{args.exp}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


    ## TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


    ## env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.exp, args.gamma) for i in range(args.num_envs)]
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    if is_continuous:
        agent = Agent_C(envs).to(dtype).to(device)
    else:
        agent = Agent_D(envs).to(dtype).to(device)


    ## Online NPG params
    extra_params = dict()
    extra_params['lr'] = args.eta
    extra_params['beta'] = args.beta
    extra_params['gamma'] = args.gamma
    extra_params['init_fisher'] = args.init_fisher
    extra_params['fisher_dtype'] = getattr(torch, args.fisher_dtype)


    ## Policy and Value Optimizer
    if is_continuous:
        optimizer = optimizers[args.optim](
            list(agent.actor_mean.parameters()) + [agent.actor_logstd],
            **extra_params)
    else:
        optimizer = optimizers[args.optim](
            list(agent.actor.parameters()),
            **extra_params)
    
    v_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=args.v_lr)


    ## ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(dtype).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(dtype).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(dtype).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(dtype).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(dtype).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(dtype).to(device)
    if is_continuous:
        action_min = torch.as_tensor(envs.action_space.low).to(dtype).to(device)
        action_max = torch.as_tensor(envs.action_space.high).to(dtype).to(device)


    ## TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(dtype).to(device)
    next_done = torch.zeros(args.num_envs).to(dtype).to(device)
    log_data = []
    save_dir = f"results/{args.env_id}/{args.optim}"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = f"{save_dir}/seed_{args.seed}.csv"

    for iteration in range(1, args.num_iterations + 1):
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            # log_entry = {"global_step": global_step}
            obs[step] = next_obs
            dones[step] = next_done

            ## ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            ## TRY NOT TO MODIFY: execute the game and log data.
            if is_continuous:
                action = action_min + (torch.tanh(action) + 1.0) * 0.5 * (action_max - action_min)  # Clipping action between max and min
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(dtype).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(dtype).to(device), torch.Tensor(next_done).to(dtype).to(device)

            if "final_info" in infos:
                return_envs = []
                length_envs = []
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                        wandb.log({
                            "Episodic/episodic_return": info["episode"]["r"],
                            "Episodic/episodic_length": info["episode"]["l"]
                        }, step=global_step)
                        return_envs.append(info["episode"]["r"].item())
                        length_envs.append(info["episode"]["l"].item())

                if return_envs:
                    episode_entry = {
                        "global_step": global_step,
                        "episodic_return": np.mean(return_envs),
                        "episodic_length": np.mean(length_envs)
                    }
                    log_data.append(episode_entry)

        ## bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(dtype).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        ## flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        ## ----------------- POLICY OPTIMIZATION -----------------
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss = -(torch.exp(newlogprob-b_logprobs[mb_inds])* mb_advantages).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Final loss
                loss = pg_loss - args.ent_coef * entropy_loss 

                ## Policy Optimization
                ratio = 1.0   # No Importance Sampling
                if args.do_importance_sampling:
                    with torch.no_grad():
                        ratio = torch.exp(newlogprob - b_logprobs[mb_inds])
                        ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                ##-----------------------------------------------------## 
                # Online NPG step: (12a), (12b), (13a), (13b)
                optimizer.zero_grad()
                optimizer.update(ratio * mb_advantages, newlogprob)
                ##-----------------------------------------------------##    


        
        ## ----------------- VALUE OPTIMIZATION -----------------
        # Calculate new minibatch size specifically for the value network
        v_minibatch_size = args.batch_size // args.v_minibatches

        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        v_lrnow = frac * args.v_lr
        v_optimizer.param_groups[0]["lr"] = v_lrnow
        
        for epoch in range(args.v_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, v_minibatch_size):
                end = start + v_minibatch_size
                mb_inds = b_inds[start:end]

                # Get only the value estimates
                newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
                
                # Value loss
                v_loss = torch.nn.functional.mse_loss(newvalue, b_returns[mb_inds])

                v_optimizer.zero_grad()
                v_loss.backward()
                # Apply gradient clipping to the critic specifically
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                v_optimizer.step()


        ## Recording results
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        ## TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        wandb.log({
            "Loss/loss": loss.item(),
            "Loss/pg_loss": pg_loss.item(),
            "Loss/v_loss": v_loss.item(),
            "Loss/entropy_loss": entropy_loss.item(),
            "Performance/advantage": b_advantages.mean(),
            "Performance/return": b_returns.mean(),
            "Performance/value": b_values.mean(),
        }, step=global_step)

        loss_entry = {
            "global_step": global_step,
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "advantage": b_advantages.mean().item(),
            "return": b_returns.mean().item(),
            "value": b_values.mean().item()
        }
        log_data.append(loss_entry)
        df = pd.DataFrame(log_data)
        df.to_csv(csv_path, index=False)


    envs.close()
    writer.close()


if __name__ == "__main__":
    main()