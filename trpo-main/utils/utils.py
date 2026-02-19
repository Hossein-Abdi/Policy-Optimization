import os
import random
import copy
import torch
import scipy
import numpy as np

from torch import nn
from torch.utils.data import Dataset
from torch.distributions.categorical import Categorical

from utils.vision_transformers import vit_tiny
from utils.resnet import ResNetImpala
from utils.convnet import NatureCNN
from utils.popart import PopArt
from utils.running_mean_std import RunningMeanStd

class MuSigmaVectorNet(nn.Module):
    def __init__(self, obs_shape, fn_embed_net, arch, dropout, hidden_size, dim_actions) -> None:
        super().__init__()
        self.mu_net = fn_embed_net(arch, dropout)
        self.shared_sigma = nn.Parameter(torch.zeros(1, dim_actions))
        self.mu_linear = nn.Linear(hidden_size, dim_actions)

    def forward(self, x):
        mu = self.mu_linear(self.mu_net(x))
        sigma = self.shared_sigma.expand_as(mu)
        return torch.cat([mu, sigma], dim=-1)

class MuSharedSigmaNet(nn.Module):
    def __init__(self, obs_shape, fn_embed_net, arch, dropout, hidden_size, dim_actions) -> None:
        super().__init__()
        self.shared_net = fn_embed_net(arch, dropout)
        self.mu_linear = nn.Linear(hidden_size, dim_actions)
        self.sigma_linear = nn.Linear(hidden_size, dim_actions)
        nn.init.constant_(self.sigma_linear.bias, 0.0)
        nn.init.constant_(self.sigma_linear.weight, 0.0) 

    def forward(self, x):
        latents = self.shared_net(x)
        mu = self.mu_linear(latents)
        sigma = self.sigma_linear(latents)

        return torch.cat([mu, sigma], dim=-1)

class MuSeparateSigmaNet(nn.Module):
    def __init__(self, obs_shape, fn_embed_net, arch, dropout, hidden_size, dim_actions) -> None:
        super().__init__()
        self.mu_net = fn_embed_net(arch, dropout)
        self.mu_linear = nn.Linear(hidden_size, dim_actions)

        self.sigma_net = fn_embed_net(arch, dropout)
        self.sigma_linear = nn.Linear(hidden_size, dim_actions)
        nn.init.constant_(self.sigma_linear.bias, 0.0)
        nn.init.constant_(self.sigma_linear.weight, 0.0) 

    def forward(self, x):
        mu = self.mu_linear(self.mu_net(x))
        sigma = self.sigma_linear(self.sigma_net(x))

        return torch.cat([mu, sigma], dim=-1)

class MuLinearSigmaNet(nn.Module):
    def __init__(self, obs_shape, fn_embed_net, arch, dropout, hidden_size, dim_actions) -> None:
        super().__init__()
        self.mu_net = fn_embed_net(arch, dropout)
        self.mu_linear = nn.Linear(hidden_size, dim_actions)

        self.sigma_linear = nn.Linear(obs_shape[0], dim_actions)
        nn.init.constant_(self.sigma_linear.bias, 0.0)
        nn.init.constant_(self.sigma_linear.weight, 0.0) # BUG should be orthogonal? nn.init.orthogonal_(self.sigma_linear.weight, gain=0.01)

    def forward(self, x):
        mu = self.mu_linear(self.mu_net(x))
        sigma = self.sigma_linear(x)

        return torch.cat([mu, sigma], dim=-1)

class AdaptiveScheduler():
    def __init__(self, kl_threshold = 0.008, min_lr=1e-6, max_lr=5e-3):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.kl_threshold = kl_threshold

    def update(self, current_lr, step_kl):
        lr = current_lr
        if step_kl > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if step_kl < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr

def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

def build_cnn(img_size, emb_size=256, device='cpu', **kwargs):
    def preprocess(obs_batch):
        obs_batch = obs_batch.astype(np.float32)
        obs_tensor = torch.from_numpy(obs_batch).to(device)
        obs = obs_tensor / 255.0
        obs = (obs - 0.5) / 0.5  # normalize to [-1, 1]
        obs = torch.permute(obs, (0, 3, 1, 2)) if len(obs.size()) == 4 else torch.permute(obs, (2, 0, 1))
        return  obs

    return lambda : NatureCNN(img_size, emb_size, **kwargs), preprocess

def build_resnet(img_size, emb_size=256, device='cpu', **kwargs):
    def preprocess(obs_batch):
        obs_batch = obs_batch.astype(np.float32)
        obs_tensor = torch.from_numpy(obs_batch).to(device)
        obs = obs_tensor / 255.0
        obs = (obs - 0.5) / 0.5  # normalize to [-1, 1]
        return torch.permute(obs, (0, 3, 1, 2)) if len(obs.size()) == 4 else torch.permute(obs, (2, 0, 1)) 

    return lambda x=None, y=None: ResNetImpala(img_size, emb_size, **kwargs), preprocess

def build_vit(img_size, emb_size=256, device='cpu', **kwargs):
    def preprocess(obs_batch):
        obs_batch = np.asarray(obs_batch).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_batch).to(device)

        obs = obs_tensor / 255.0
        obs = (obs - 0.5) / 0.5  # normalize to [-1, 1]
        obs = torch.permute(obs, (0, 3, 1, 2)) if len(obs.size()) == 4 else torch.permute(obs, (2, 0, 1))
        return  obs

    return lambda : vit_tiny(img_size=img_size, embed_dim=emb_size, **kwargs), preprocess

def build_mlp(obs_space, hidden_size=None, num_layers=None, p_dropout=0.0, activation_fn=nn.ReLU, device='cuda'):
    def preprocess(obs_batch):
        obs_batch = np.asarray(obs_batch).astype(np.float32)
        return torch.from_numpy(obs_batch).to(device)

    input_dim = obs_space.shape[0]

    def fn_neural_net(net_arch=None, p_dropout=0.0):
        if net_arch is None:
            assert hidden_size is not None and num_layers is not None
            net_arch = [hidden_size] * num_layers

        modules = []
        if len(net_arch) > 0:
            modules.append(nn.Linear(input_dim, net_arch[0], bias=True))
            # if p_dropout > 0:
            #     modules.append(nn.Dropout(p_dropout))
            modules.append(activation_fn())
        
        for idx in range(len(net_arch) - 1):
            # if p_dropout > 0:
            #     modules.append(nn.Dropout(p_dropout))
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=True))
            modules.append(activation_fn())

        if p_dropout > 0:
            modules.append(nn.Dropout(p_dropout))

        return nn.Sequential(*modules)

    return fn_neural_net, preprocess

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

class BufferDataset(Dataset):
    def __init__(self, obs, ret, logp):
        self.obs = obs
        self.ret = ret
        self.logp = logp

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):
        return self.obs[index], self.ret[index], self.logp[index]

class TrajReplayBuffer:
    """
    A simple FIFO experience replay buffer
    """
    def __init__(self, obs_shape, act_shape, size, nsteps, num_envs=1):
        fixed_shape = (size, nsteps)
        self.obs_buf = torch.zeros(fixed_shape + obs_shape, dtype=torch.float32)
        self.rew_buf = torch.zeros(fixed_shape, dtype=torch.float32)
        self.act_buf = torch.zeros(fixed_shape, dtype=torch.int64)
        self.val_buf = torch.zeros(fixed_shape, dtype=torch.float32)
        self.n_val_buf = torch.zeros(fixed_shape, dtype=torch.float32)
        self.gae_buf = torch.zeros(fixed_shape, dtype=torch.float32)
        self.logits_buf = torch.zeros(fixed_shape + act_shape, dtype=torch.float32)
        self.done_buf = torch.zeros(fixed_shape, dtype=torch.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.num_envs = num_envs

    def store(self, obs, rew, act, val, n_val, gae, logits, done):
        # store all on RAM, not GPU
        self.obs_buf[self.ptr : self.ptr + self.num_envs] = obs.cpu()
        self.rew_buf[self.ptr: self.ptr + self.num_envs] = rew.cpu()
        self.act_buf[self.ptr: self.ptr + self.num_envs] = act.cpu()
        self.val_buf[self.ptr: self.ptr + self.num_envs] = val.cpu()
        self.n_val_buf[self.ptr: self.ptr + self.num_envs] = n_val.cpu()
        self.gae_buf[self.ptr: self.ptr + self.num_envs] = gae.cpu()
        self.logits_buf[self.ptr: self.ptr + self.num_envs] = logits.cpu()
        self.done_buf[self.ptr: self.ptr + self.num_envs] = done.cpu()
        self.ptr = (self.ptr+self.num_envs) % self.max_size
        self.size = min(self.size+self.num_envs, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     rew=self.rew_buf[idxs],
                     act=self.act_buf[idxs],
                     val=self.val_buf[idxs],
                     n_val=self.n_val_buf[idxs],
                     gae=self.gae_buf[idxs],
                     logits=self.logits_buf[idxs],
                     done=self.done_buf[idxs])
        return batch

class TrajAdvReplayBuffer:
    """
    A simple FIFO experience replay buffer
    """
    def __init__(self, obs_shape, act_shape, size, nsteps, num_envs=1):
        fixed_shape = (size, nsteps)
        self.obs_buf = torch.zeros(fixed_shape + obs_shape, dtype=torch.float32)
        self.rew_buf = torch.zeros(fixed_shape, dtype=torch.float32)
        self.act_buf = torch.zeros(fixed_shape, dtype=torch.int64)
        self.logits_buf = torch.zeros(fixed_shape + act_shape, dtype=torch.float32)
        self.done_buf = torch.zeros(fixed_shape, dtype=torch.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.num_envs = num_envs

    def store(self, obs, rew, act, logits, done):
        # store all on RAM, not GPU
        self.obs_buf[self.ptr : self.ptr + self.num_envs] = obs.cpu()
        self.rew_buf[self.ptr: self.ptr + self.num_envs] = rew.cpu()
        self.act_buf[self.ptr: self.ptr + self.num_envs] = act.cpu()
        self.logits_buf[self.ptr: self.ptr + self.num_envs] = logits.cpu()
        self.done_buf[self.ptr: self.ptr + self.num_envs] = done.cpu()
        self.ptr = (self.ptr+self.num_envs) % self.max_size
        self.size = min(self.size+self.num_envs, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     rew=self.rew_buf[idxs],
                     act=self.act_buf[idxs],
                     logits=self.logits_buf[idxs],
                     done=self.done_buf[idxs])
        return batch

class CategoricalActor(nn.Module):
    def __init__(self, embed_net, embed_dim, n_actions, feature_input=False):
        super().__init__()
        self.n_actions = n_actions # used for one-hot encoding
        if feature_input:
            self.embed_net = lambda x: torch.flatten(embed_net.forward_features(x), start_dim=1)
        else:
            self.embed_net = embed_net

        self.logits_net = nn.Sequential(
            nn.Linear(embed_dim, n_actions),
        )

    def forward(self, obs):
        latents = self.embed_net(obs)
        logits = self.logits_net(latents)
        return logits

class ValueCritic(nn.Module):
    def __init__(self, embed_net, embed_dim, n_actions, feature_input=False):
        super().__init__()
        self.n_actions = n_actions # used for one-hot encoding
        if feature_input:
            self.embed_net = lambda x: torch.flatten(embed_net.forward_features(x), start_dim=1)
        else:
            self.embed_net = embed_net

        self.v_net = nn.Sequential(
            # nn.Linear(embed_dim, embed_dim),
            # nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, obs):
        latents = self.embed_net(obs)
        vals = self.v_net(latents)
        return vals.squeeze(-1)

class SeparateActorCritic(nn.Module):
    def __init__(self, embed_net, embed_dim, n_actions, feature_input=False):
        super().__init__()
        self.is_discrete = n_actions is not None
        self.logits_net = CategoricalActor(embed_net, embed_dim, n_actions, feature_input=False)
        self.v_net = ValueCritic(copy.deepcopy(embed_net), embed_dim, n_actions, feature_input=False)

    def forward(self, obs):
        logits = self.logits_net(obs)
        vals = self.v_net(obs)
        return vals, logits
    
    def forward_logits(self, obs):
        return self.logits_net(obs)
    
    def forward_vals(self, obs):
        return self.v_net(obs)

class PhasicActorCritic(nn.Module):
    def __init__(self, embed_net, embed_dim, n_actions, feature_input=False):
        super().__init__()
        self.logits_net = CategoricalActor(embed_net, embed_dim, n_actions, feature_input=False)
        self.pi_v_net = ValueCritic(embed_net, embed_dim, n_actions, feature_input=False)
        self.v_net = ValueCritic(copy.deepcopy(embed_net), embed_dim, n_actions, feature_input=False)

    def forward(self, obs):
        logits = self.logits_net(obs)
        vals = self.v_net(obs)
        return vals, logits

    def forward_full(self, obs):
        logits = self.logits_net(obs)
        pi_vals = self.pi_v_net(obs)
        vals = self.v_net(obs)
        return vals, pi_vals, logits


class ActorCritic(nn.Module):
    def __init__(self, fn_embed_net, obs_shape, nets_config, n_actions=None, 
                 dim_actions=None, with_popart=True, sigma_type='vector', device=None):
                 # sigma_type: 's_shared', 'mu_shared', 'separate', 'linear

        super().__init__()
        self.n_actions = n_actions # used for one-hot encoding
        self.obs_rms = RunningMeanStd(obs_shape).to(device) if nets_config.norm_obs else None
        self.a_dropout = nets_config.a_dropout
        self.c_dropout = nets_config.c_dropout

        self.with_popart = with_popart
        self.sigma_type = sigma_type
        self.is_discrete = n_actions is not None

        actor_arch = [nets_config.a_hidden_size] * nets_config.a_num_layers
        critic_arch = [nets_config.c_hidden_size] * nets_config.c_num_layers

        if isinstance(fn_embed_net(actor_arch, self.a_dropout), (ResNetImpala, NatureCNN) ): # hardcoded
            # only for discrete actions
            assert n_actions is not None

            self.pi_net = nn.Sequential(
                fn_embed_net(actor_arch, self.a_dropout),
                nn.Linear(nets_config.a_hidden_size, n_actions),
            )
        else:
            if n_actions is not None:
                self.pi_net = nn.Sequential(
                    fn_embed_net(actor_arch, self.a_dropout),
                    nn.Linear(nets_config.a_hidden_size, n_actions),
                )
            else:
                assert dim_actions is not None
                if sigma_type == 'vector':
                    self.pi_net = MuSigmaVectorNet(obs_shape, fn_embed_net, actor_arch, nets_config.a_dropout, nets_config.a_hidden_size, dim_actions)
                elif sigma_type == 'mu_shared':
                    self.pi_net = MuSharedSigmaNet(obs_shape, fn_embed_net, actor_arch, nets_config.a_dropout, nets_config.a_hidden_size, dim_actions)
                elif sigma_type == 'separate':
                    self.pi_net = MuSeparateSigmaNet(obs_shape, fn_embed_net, actor_arch, nets_config.a_dropout, nets_config.a_hidden_size, dim_actions)
                elif sigma_type == 'linear':
                    self.pi_net = MuLinearSigmaNet(obs_shape, fn_embed_net, actor_arch, nets_config.a_dropout, nets_config.a_hidden_size, dim_actions)

        self.last_v_layer = PopArt(nets_config.c_hidden_size, 1, norm_axes=0) if with_popart else nn.Linear(nets_config.c_hidden_size, 1)
        self.v_net = nn.Sequential(
            fn_embed_net(critic_arch, self.c_dropout), 
            self.last_v_layer,
        )
    
    def forward(self, obs):
        outputs = self.forward_pi(obs)
        vals = self.forward_v(obs)
        return vals, outputs

    def forward_pi(self, obs):
        outputs = self.pi_net(obs)
        return outputs

    def forward_v(self, obs):
        vals = self.v_net(obs)
        return vals.squeeze(-1)


class SharedActorCritic(nn.Module):
    def __init__(self, fn_embed_net, obs_shape, nets_config, n_actions=None, 
                 dim_actions=None, with_popart=True, sigma_type='vector', device=None):
                 # sigma_type: 's_shared', 'mu_shared', 'separate', 'linear

        super().__init__()
        self.n_actions = n_actions # used for one-hot encoding
        self.obs_rms = RunningMeanStd(obs_shape).to(device) if nets_config.norm_obs else None
        self.dropout = nets_config.dropout

        self.with_popart = with_popart
        self.sigma_type = sigma_type
        assert sigma_type == 'vector', 'Only vector sigma_type is supported currently'
        self.is_discrete = n_actions is not None

        self.backbone_net = fn_embed_net()

        if n_actions is not None:
            self.pi_head = nn.Linear(nets_config.hidden_size, n_actions)
            self.shared_sigma = None
        else:
            assert dim_actions is not None
            self.pi_head = nn.Linear(nets_config.hidden_size, dim_actions)
            self.shared_sigma = nn.Parameter(torch.zeros(1, dim_actions))

        self.last_v_layer = PopArt(nets_config.hidden_size, 1, norm_axes=0) if with_popart else nn.Linear(nets_config.hidden_size, 1)
    
    def forward(self, obs):
        latents = self.backbone_net(obs)
        outputs = self.forward_pi(latents=latents)
        vals = self.forward_v(latents=latents)
        return vals, outputs

    def forward_pi(self, obs=None, latents=None):
        if latents is not None:
            mu = self.pi_head(latents)
        else:
            latents = self.backbone_net(obs)
            mu = self.pi_head(latents)
        sigma = self.shared_sigma.expand_as(mu) if self.shared_sigma is not None else None
        outputs = torch.cat([mu, sigma], dim=-1) if sigma is not None else mu
        return outputs

    def forward_v(self, obs=None, latents=None):
        if latents is not None:
            vals = self.last_v_layer(latents)
        else:
            latents = self.backbone_net(obs)
            vals = self.last_v_layer(latents)
        return vals.squeeze(-1)


def model_step(model, obs, deterministic=False):
    with torch.no_grad():
        obs = model.obs_rms(obs) if model.obs_rms is not None else obs
        vals, outputs = model(obs)
        if hasattr(model, 'last_v_layer') and hasattr(model.last_v_layer, 'unnormalize'): 
            vals = model.last_v_layer.unnormalize(vals) # obtain unnormalized

        if model.is_discrete:
            if deterministic: 
                act = torch.argmax(outputs, dim=-1)
            else:
                pi = Categorical(logits=outputs)
                act = pi.sample()
        else:
            mean, logstd = outputs.chunk(2, dim=-1)
            if deterministic:
                act = mean
            else:
                act = mean + torch.exp(logstd) * torch.randn_like(mean)

    return act, vals, outputs

def gpt_model_step(model, mb_obs, mb_dones, mb_tel, deterministic=False):
    with torch.no_grad():
        last_obs = mb_obs[-1].unsqueeze(1)
        seq_dones = torch.from_numpy(np.asarray(mb_dones).astype(np.float32)).transpose(0, 1).to(last_obs.device)
        seq_id = (torch.cumsum(seq_dones, dim=-1) - seq_dones).long()
        same_seq_tag = seq_id.unsqueeze(-2) == seq_id.unsqueeze(-1)

        seq_tel = torch.stack(mb_tel, dim=0).transpose(0, 1).to(last_obs.device)

        last_id, last_tel, last_mask = seq_id[:, -1:], seq_tel[:, -1:], same_seq_tag[:, -1:]
        last_obs = model.obs_rms(last_obs) if model.obs_rms is not None else last_obs # normalize obs
        all_outputs = model(last_obs, last_id, seq_tel=last_tel, attn_mask=last_mask, use_kv_cache=True)
        
        if model.is_discrete:
            outputs = all_outputs[:, -1]
            act = torch.argmax(outputs, dim=-1) if deterministic else Categorical(logits=outputs).sample()
        else:
            mean, logstd = all_outputs.chunk(2, dim=-1)
            mean, logstd = mean[:, -1], logstd[:, -1]
            outputs = torch.cat([mean, logstd], dim=-1)
            act = mean if deterministic else mean + torch.exp(logstd) * torch.randn_like(mean)

    return act, None, outputs

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(params, flat_params):
    prev_ind = 0
    for param in params:
        flat_size = torch.numel(param)
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def set_grads_from_flat(params, flat_grads):
    prev_ind = 0
    for param in params:
        if param.grad is None:
            continue
        flat_size = torch.numel(param.grad)
        param.grad.data.copy_(
            flat_grads[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size