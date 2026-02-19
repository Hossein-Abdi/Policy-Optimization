import itertools

def generate_procgen_tasks():
    with open('tasks.txt', 'w+') as fout:
        # config_list = ['ppo_resnet_shared.yaml', 'kfac_resnet_shared.yaml']
        # config_list = ['adv_resnet_shared.yaml', 'ppo_resnet_shared.yaml', 'kfac_resnet_shared.yaml']
        config_list = ['adv_resnet_shared.yaml']
        # env_list = ['bigfish', 'caveflyer', 'coinrun', 'fruitbot', 'jumper', 'ninja', 'plunder', 'starpilot']
        # env_list = ['caveflyer', 'coinrun', 'dodgeball', 'starpilot']
        # env_list = ['coinrun', 'starpilot', 'caveflyer', 'dodgeball', 'fruitbot', 
        #             'chaser', 'miner', 'jumper', 'leaper', 'maze', 'bigfish', 
        #             'heist', 'climber', 'plunder', 'ninja', 'bossfight'
        #             ]
        # env_list = ['coinrun', 'starpilot', 'caveflyer', 
        #             'miner', 'jumper', 'maze', 'bossfight'
        #             ]
        env_list = ['bigfish']
        # env_list = ['atari.BeamRiderNoFrameskip-v4', 'atari.BreakoutNoFrameskip-v4', 'atari.EnduroNoFrameskip-v4', 'atari.PongNoFrameskip-v4', 'atari.QbertNoFrameskip-v4', 'atari.SeaquestNoFrameskip-v4', 'atari.SpaceInvadersNoFrameskip-v4']
        dist_list = ['easy']
        start_levels = [0]
        num_levels = [10]
        device_list = [0, 1]

        devices = itertools.cycle(device_list)

        for _ in range(5):
            all_tasks = itertools.product(config_list, env_list, dist_list, start_levels, num_levels)
            for t in all_tasks:
                fout.write(f'{t[0]}\t{t[1]}-{t[2]}-{t[3]}-{t[4]}\t{next(devices)}\n')


def generate_atari_tasks():
    with open('tasks.txt', 'w+') as fout:
        # config_list = ['ppo_resnet_shared.yaml', 'kfac_resnet_shared.yaml']
        # config_list = ['kfac_cnn_shared.yaml']
        config_list = ['ppo_resnet_shared.yaml', 'kfac_resnet_shared.yaml']
        env_list = ['atari.BeamRiderNoFrameskip-v4', 'atari.EnduroNoFrameskip-v4', 'atari.PongNoFrameskip-v4', 'atari.QbertNoFrameskip-v4', 'atari.SeaquestNoFrameskip-v4', 'atari.SpaceInvadersNoFrameskip-v4']
        device_list = [0, 1]

        devices = itertools.cycle(device_list)

        for _ in range(5):
            all_tasks = itertools.product(config_list, env_list)
            for t in all_tasks:
                fout.write(f'{t[0]}\t{t[1]}\t{next(devices)}\n')


def generate_mujoco_tasks():
    with open('tasks.txt', 'w+') as fout:
        # config_list = ['empirical_mlp.yaml', 'true_mlp.yaml']
        config_list = ['true_mlp.yaml']
        env_list = ['swimmer', 'halfcheetah', 'hopper', 'walker2d', 'humanoid', 'ant', 'humanoidstandup']
        # env_list = ['swimmer', 'halfcheetah', 'hopper', 'walker2d', 'humanoidstandup']
        # env_list = ['humanoid']
        # sigma_types = ['vector', 'mu_shared', 'separate', 'linear']
        sigma_types = ['vector']
        pi_epochs = [1]
        cg_steps = [10]
        device_list = [0, 1]

        devices = itertools.cycle(device_list)

        for _ in range(5):
            all_tasks = itertools.product(config_list, env_list, sigma_types, pi_epochs, cg_steps)
            for t in all_tasks:
                fout.write(f'{t[0]}\t{t[1]}\t{t[2]}\t{t[3]}\t{t[4]}\t{next(devices)}\n')

if __name__ == '__main__':
    # generate_atari_tasks()
    # generate_procgen_tasks()
    generate_mujoco_tasks()
