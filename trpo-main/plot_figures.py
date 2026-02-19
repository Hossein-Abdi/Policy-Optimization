import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

import seaborn # sets some style parameters automatically

# COLORS = [(57, 106, 177), (218, 124, 48)] 
COLORS = seaborn.color_palette('colorblind')
matplotlib.rcParams.update({'errorbar.capsize': 2})

matplotlib.rcParams["text.usetex"] = True

envs_list = ['invertedpendulum', 'hopper', 'halfcheetah', 'walker2d'] 
action_dims = {'invertedpendulum': 1, 'hopper': 3, 'halfcheetah': 6, 'walker2d': 6, 'humanoid': 17}

def load_event_data(scalar_tag, events_dir, keys_mapping, first_epochs=306, divide_1k=False):
    # === List all TensorBoard event files ===
    event_files = []
    for dirpath, _, filenames in os.walk(events_dir):
        for filename in filenames:
            if filename.startswith("events.out.tfevents"):
                full_path = os.path.join(dirpath, filename)
                event_files.append(full_path)

    tags_to_plot = sorted(keys_mapping.keys(), reverse=True)

    tagged_event_files = {}
    for event_file in event_files:
        for tag in tags_to_plot:
            if tag in event_file:
                if tag not in tagged_event_files:
                    tagged_event_files[tag] = []
                tagged_event_files[tag].append(event_file)


    if len(tagged_event_files) == 0:
        print("No event files found with the specified tags.")
        exit(1)

    kw_epochs = {}
    kw_eprewmean = {}

    for tag in tagged_event_files.keys():
        event_file_paths = tagged_event_files[tag]
        for event_file_path in event_file_paths:
            # print(f"Loading event file: {event_file_path}")
            ea = event_accumulator.EventAccumulator(event_file_path)
            ea.Reload()

            scalar_tags = ea.Tags()['scalars']
            # print(f"Available scalar tags: {scalar_tags}")

            if scalar_tag not in scalar_tags:
                print(f"Scalar tag '{scalar_tag}' not found in event file.")
                continue

            # Get the scalar values for the specified tag
            events = ea.Scalars(scalar_tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            if divide_1k: 
                values = values / 1000.0

            if first_epochs is not None and (len(steps) < first_epochs or len(values) < first_epochs):
                print(f"Warning: Event file '{event_file_path}' has only {len(steps)} epochs, which is less than the required {first_epochs} epochs. Skipping this file.")
                continue

            if tag not in kw_epochs:
                kw_epochs[tag] = []
            kw_epochs[tag].append(steps[:first_epochs])
            if tag not in kw_eprewmean:
                kw_eprewmean[tag] = []
            kw_eprewmean[tag].append(values[:first_epochs])
    
    return kw_epochs, kw_eprewmean, tags_to_plot

def make_bar_plots(ax, title, iter_to_key, kw_epochs, kw_eprewmean, tags_to_plot, keys_mapping, color_idx=0):
    data_to_plot = []
    x_labels = []
    for k, v in iter_to_key.items():
        try:
            values = np.stack(kw_eprewmean[v], axis=0)
        except (ValueError, KeyError):
            breakpoint()

        y_mean = np.mean(values, axis=0)
        y_stderr = np.std(values, axis=0) / np.sqrt(len(values))

        data_to_plot.append([y_mean[-1], y_stderr[-1]])
        x_labels.append(k)

    data_to_plot = np.array(data_to_plot)

    ax.bar(x_labels, data_to_plot[:, 0], color=COLORS[color_idx], width=0.4, edgecolor='black', yerr=data_to_plot[:, 1])
    # ax.errorbar(x_labels, data_to_plot[:, 0], data_to_plot[:, 1], color='red')
    ax.plot(x_labels, data_to_plot[:, 0], color='red')

def make_line_plots_dropout():
    title = 'dropout-mlp-empirical-no-damping'
    scalar_tag = 'train/eprewmean'
    events_dir = "log_backup/0612_mlp_dropout/"
    # action dims: invertedpendulum(1), hopper(3), halfcheetah(6), walker2d(6)
    dimx, dimy = (int(np.ceil(len(envs_list)/4)), 4)
    fig, axarr = plt.subplots(dimx, dimy, figsize=(dimy * 3.5, dimx * 3.5), dpi=300)

    dropout_list = [0.0, 0.1, 0.2, 0.3]
    for i, env in enumerate(envs_list):
        ax = axarr[i // dimy][i % dimy] if dimx > 1 else axarr[i % dimy]
        ax.grid(True)
        ax.set_title(f'Action dim: {action_dims[env]} - {env}')
        ax.set_xlabel("Dropout rate")
        if i == 0:
            ax.set_ylabel("Episodic return")

        eprewmean = {}
        eprewstderr = {}
        for dropout in dropout_list:
            keys_mapping = {
                            f'empirical.mlp.dropout_{dropout}.linear_1.0.damping_0.0': 'Empirical', 
                            f'empirical.mlp.dropout_{dropout}.linear_0.0.damping_0.0': 'True', 
                            }
            new_keys_mapping = {f'{k}/{env}': v for k, v in keys_mapping.items()}
            _, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping)

            for key in tags_to_plot:
                values = np.stack(kw_eprewmean[key], axis=0)
                y_mean_all = np.mean(values, axis=0)
                y_stderr_all = np.std(values, axis=0) / np.sqrt(len(values))
                y_mean = y_mean_all[-1]  # Get the last value for each dropout
                y_stderr = y_stderr_all[-1]  # Get the last stderr for each dropout

                mapped_key = new_keys_mapping[key]
                if mapped_key not in eprewmean:
                    eprewmean[mapped_key] = []
                    eprewstderr[mapped_key] = []

                eprewmean[mapped_key].append(y_mean)
                eprewstderr[mapped_key].append(y_stderr)

        color_idx = 0
        for key in eprewmean.keys():
            # ax.plot(dropout_list, eprewmean[key], label=key, color=COLORS[color_idx], linestyle='solid', marker='D', linewidth=1.0, rasterized=True)
            # ax.fill_between(dropout_list, np.array(eprewmean[key]) - np.array(eprewstderr[key]), np.array(eprewmean[key]) + np.array(eprewstderr[key]), color=COLORS[color_idx], alpha=.25, linewidth=0.0, rasterized=True)
            ax.errorbar(dropout_list, eprewmean[key], yerr=[eprewstderr[key], eprewstderr[key]], label=key, color=COLORS[color_idx], fmt='o', linestyle='solid', linewidth=2.0, rasterized=True)
            color_idx += 1

    ax.legend()

    # === Plot and save each scalar tag ===
    output_dir = "figs"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")

    ax0 = fig.add_subplot(111, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("\nAll plots saved to:", title)

def make_training_plot():
    # === Configuration ===
    for dropout in [0.0, 0.1, 0.2, 0.3]:
        title = f'empirical-mlp-training-{dropout}'
        scalar_tag = 'train/eprewmean'
        events_dir = "log_backup/0612_mlp_dropout/"

        dimx, dimy = (int(np.ceil(len(envs_list)/4)), 4)
        keys_mapping = {
                        f'empirical.mlp.dropout_{dropout}.linear_1.0.damping_0.0': 'Empirical', 
                        f'empirical.mlp.dropout_{dropout}.linear_0.0.damping_0.0': 'True', 
                        }
        fig, axarr = plt.subplots(dimx, dimy, figsize=(dimy * 3.5, dimx * 3.5), dpi=300)

        for i, env in enumerate(envs_list):
            ax = axarr[i // dimy][i % dimy] if dimx > 1 else axarr[i % dimy]
            ax.set_title(f'Action dim: {action_dims[env]} - {env}')
            ax.set_xlabel("Epoch")
            if i == 0:
                ax.set_ylabel("Episodic return")
            # ax.set_ylim(-1000, 5000)
            # ax.set_xlim(0, 1000)
            new_keys_mapping = {f'{k}/{env}': v for k, v in keys_mapping.items()}

            kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping)

            color_idx = 0
            for key in tags_to_plot:
                try:
                    steps = np.stack(kw_epochs[key], axis=0)
                except ValueError:
                    breakpoint()
                values = np.stack(kw_eprewmean[key], axis=0)

                x = np.mean(steps, axis=0)
                y_mean = np.mean(values, axis=0)
                y_stderr = np.std(values, axis=0) / np.sqrt(len(values))

                ax.plot(x, y_mean, label=new_keys_mapping[key], color=COLORS[color_idx], linestyle='solid', linewidth=2.0, rasterized=True)
                ax.fill_between(x, y_mean - y_stderr, y_mean + y_stderr, color=COLORS[color_idx], alpha=.4, linewidth=0.0, rasterized=True)
                color_idx += 1

            # plt.grid(True)
            ax.legend()

        # === Plot and save each scalar tag ===
        output_dir = "figs"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{title}.pdf")

        ax0 = fig.add_subplot(111, frame_on=False)
        ax0.set_xticks([])
        ax0.set_yticks([])

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print("\nAll plots saved to:", title)


def make_humanoid_iters():
    from collections import OrderedDict

    # === Configuration ===
    num_iters = [1, 2, 4, 8]
    dropout_list = [0.0, 0.1, 0.2, 0.3]
    title = f'empirical-mlp-training-humanoid'
    dimx, dimy = (int(np.ceil(len(dropout_list)/4)), 4)
    fig, axarr = plt.subplots(dimx, dimy, figsize=(dimy * 3.5, dimx * 3.5), dpi=300, sharey=True)

    for j, dropout in enumerate(dropout_list):
        scalar_tag = 'train/eprewmean'
        events_dir = "log_backup/0612_mlp_dropout/"

        ax = axarr[j // dimy][j % dimy] if dimx > 1 else axarr[j % dimy]
        ax.set_title(f"Action dim: {action_dims['humanoid']} - Dropout rate: {dropout}")
        ax.set_xlabel("Off-policy iterations")
        if j == 0:
            ax.set_ylabel("Episodic return")

        bar_data_mean = OrderedDict()
        bar_data_stderr = OrderedDict()

        for i in num_iters:
            keys_mapping = {f'empirical.mlp{f"_{i}epoch" if i >1 else ""}.dropout_{dropout}.linear_0.0.damping_0.0': f'True', 
                            f'empirical.mlp{f"_{i}epoch" if i >1 else ""}.dropout_{dropout}.linear_1.0.damping_0.0': f'Empirical', 
                            }
            # ax.set_ylim(-1000, 5000)
            # ax.set_xlim(0, 4000)
            new_keys_mapping = {f'{k}/humanoid': v for k, v in keys_mapping.items()}

            kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping)

            for key in tags_to_plot:
                new_key = new_keys_mapping[key]
                if bar_data_mean.get(new_key) is None:
                    bar_data_mean[new_key] = []
                    bar_data_stderr[new_key] = []

                try:
                    steps = np.stack(kw_epochs[key], axis=0)
                except ValueError:
                    breakpoint()
                values = np.stack(kw_eprewmean[key], axis=0)

                x = np.mean(steps, axis=0)
                y_mean = np.mean(values, axis=0)
                y_stderr = np.std(values, axis=0) / np.sqrt(len(values))

                bar_data_mean[new_key].append(y_mean[-1])  # Get the last value for each iteration
                bar_data_stderr[new_key].append(y_stderr[-1])  # Get the last stderr for each iteration

        color_idx = 0
        width = 0.35  # the width of the bars
        ind = np.arange(len(num_iters))  # the x locations for the groups

        y_mean = bar_data_mean['Empirical']
        y_stderr = bar_data_stderr['Empirical']
        rects1 = ax.bar(ind - width/2, y_mean, width, yerr=y_stderr, color=COLORS[color_idx], label='Empirical')
        color_idx += 1

        y_mean = bar_data_mean['True']
        y_stderr = bar_data_stderr['True']
        rects2 = ax.bar(ind + width/2, y_mean, width, yerr=y_stderr, color=COLORS[color_idx], label='True')
        color_idx += 1

        # ax.plot(x, y_mean, label=new_keys_mapping[key], color=COLORS[color_idx], linestyle='solid', linewidth=2.0, rasterized=True)
        # ax.fill_between(x, y_mean - y_stderr, y_mean + y_stderr, color=COLORS[color_idx], alpha=.25, linewidth=0.0, rasterized=True)

        # plt.grid(True)
        ax.legend()
        ax.set_xticks(ind)
        ax.set_xticklabels(num_iters)

    # === Plot and save each scalar tag ===
    output_dir = "figs"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")

    ax0 = fig.add_subplot(111, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("\nAll plots saved to:", title)


def make_training_plot_cnn():
    # === Configuration ===
    envs_list = ['starpilot', 'dodgeball', 'bigfish', 
                'climber', 'plunder']
    for dropout in [0.2]:
        title = f'empirical-cnn-training-{dropout}'
        scalar_tag = 'train/eprewmean'
        events_dir = "log_backup/0714_procgen_cnn/"

        dimx, dimy = (int(np.ceil(len(envs_list)/5)), 5)
        keys_mapping = {
                        f'empirical.cnn_bn_2epoch.dropout_{dropout}.linear_1.0.damping_0.0.lr_pi_0.1': 'Empirical', 
                        f'empirical.cnn_bn_2epoch.dropout_{dropout}.linear_0.0.damping_0.0.lr_pi_0.1': 'True', 
                        }
        fig, axarr = plt.subplots(dimx, dimy, figsize=(dimy * 3.5, dimx * 3.5), dpi=300)

        for i, env in enumerate(envs_list):
            ax = axarr[i // dimy][i % dimy] if dimx > 1 else axarr[i % dimy]
            ax.set_title(f'{env}')
            ax.set_xlabel("Epoch")
            if i == 0:
                ax.set_ylabel("Episodic return")
            # ax.set_ylim(-1000, 5000)
            # ax.set_xlim(0, 1000)
            new_keys_mapping = {f'{k}/{env}': v for k, v in keys_mapping.items()}

            kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping)

            color_idx = 0
            for key in tags_to_plot:
                try:
                    steps = np.stack(kw_epochs[key], axis=0)
                except ValueError:
                    breakpoint()
                values = np.stack(kw_eprewmean[key], axis=0)

                x = np.mean(steps, axis=0)
                y_mean = np.mean(values, axis=0)
                y_stderr = np.std(values, axis=0) / np.sqrt(len(values))

                ax.plot(x, y_mean, label=new_keys_mapping[key], color=COLORS[color_idx], linestyle='solid', linewidth=2.0, rasterized=True)
                ax.fill_between(x, y_mean - y_stderr, y_mean + y_stderr, color=COLORS[color_idx], alpha=.4, linewidth=0.0, rasterized=True)
                color_idx += 1

            # plt.grid(True)
            ax.legend()

        # === Plot and save each scalar tag ===
        output_dir = "figs"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{title}.pdf")

        ax0 = fig.add_subplot(111, frame_on=False)
        ax0.set_xticks([])
        ax0.set_yticks([])

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print("\nAll plots saved to:", title)

def make_mujoco_separate():
    title = f'mujoco-separate'
    scalar_tag = 'train/eprewmean'
    events_dir = "plotting_logs/0110_10M_all/"
    envs_list = ['halfcheetah', 'ant', 'humanoid', 'humanoidstandup']
    dimx, dimy = (int(np.ceil(len(envs_list)/4)), 4)
    keys_mapping = {
        f'adv.karzmarz_False.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': 'RAT',
        f'diag.karzmarz_False.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_l2_clip_0.5.vector.damping_0.1.lr_pi_0.05': 'Sophia',
        f'kfac.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.01': 'KFAC',
        f'true.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': 'FVP+CG',
    }
    fig, axarr = plt.subplots(dimx, dimy, figsize=(dimy * 3.5, dimx * 3.5), dpi=300)
    from plot_tb_events import make_plots

    for i, env in enumerate(envs_list):
        ax = axarr[i // dimy][i % dimy] if dimx > 1 else axarr[i % dimy]
        title_name = env.split('-')[0] if '-' in env else env
        ax.set_title(title_name)
        ax.set_xlabel("Epoch")
        if i % 4 == 0:
            ax.set_ylabel("Episodic return")
        # ax.set_ylim(-1000, 5000)
        # ax.set_xlim(0, 1000)
        new_keys_mapping = {f'{k}/{env}.': v for k, v in keys_mapping.items()}

        kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping, first_epochs=None)
        make_plots(ax, env, kw_epochs, kw_eprewmean, tags_to_plot, new_keys_mapping)

    # === Plot and save each scalar tag ===
    output_dir = "tensorboard_plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")

    ax0 = fig.add_subplot(111, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    plt.tight_layout()
    output_dir = "figs"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")
    plt.savefig(filename)
    plt.close()
    print("\nAll plots saved to:", title)

def make_mujoco_shared():
    title = f'mujoco-shared'
    scalar_tag = 'train/eprewmean'
    events_dir = "plotting_logs/0110_shared_fixed_adv/"
    envs_list = ['swimmer', 'halfcheetah', 'ant', 'hopper', 'walker2d', 'humanoid', 'humanoidstandup']
    dimx, dimy = (int(np.ceil(len(envs_list)/4)), 4)
    keys_mapping = {
        f'shared.adv.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.1.lr_0.05': 'RAT',
        f'shared.kfac.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.1.lr_0.001': 'ACKTR',
        f'shared.ppo.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.1.lr_0.001': 'PPO',
        f'shared.diag.karzmarz_False.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.1.lr_0.05': 'Sophia',
    }
    fig, axarr = plt.subplots(dimx, dimy, figsize=(dimy * 3.5, dimx * 3.5), dpi=300)
    from plot_tb_events import make_plots

    for i, env in enumerate(envs_list):
        ax = axarr[i // dimy][i % dimy] if dimx > 1 else axarr[i % dimy]
        title_name = env.split('-')[0] if '-' in env else env
        ax.set_title(title_name)
        ax.set_xlabel("Epoch")
        if i % 4 == 0:
            ax.set_ylabel("Episodic return")
        # ax.set_ylim(-1000, 5000)
        # ax.set_xlim(0, 1000)
        new_keys_mapping = {f'{k}/{env}.': v for k, v in keys_mapping.items()}

        kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping, first_epochs=610)
        make_plots(ax, env, kw_epochs, kw_eprewmean, tags_to_plot, new_keys_mapping)

    # === Plot and save each scalar tag ===
    output_dir = "tensorboard_plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")

    ax0 = fig.add_subplot(111, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    plt.tight_layout()
    output_dir = "figs"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")
    plt.savefig(filename)
    plt.close()
    print("\nAll plots saved to:", title)

def make_procgen_shared():
    title = f'procgen'
    scalar_tag = 'train/eprewmean'
    events_dir = "plotting_logs/0120_procgen_6M/"
    envs_list = ['bigfish-easy-0-10', 'bossfight-easy-0-10', 'caveflyer-easy-0-10', 
                 'coinrun-easy-0-10', 'jumper-easy-0-10', 'maze-easy-0-10',
                 'miner-easy-0-10', 'starpilot-easy-0-10']
    dimx, dimy = (int(np.ceil(len(envs_list)/4)), 4)
    keys_mapping = {
        f'shared.adv.resnet.dropout_0.0.damping_0.5.lr_0.5': 'RAT',
        f'shared.kfac.resnet.dropout_0.0.damping_0.1.lr_0.0005': 'ACKTR',
        f'shared.ppo.resnet.dropout_0.0.damping_0.1.lr_0.001': 'PPO',
    }
    fig, axarr = plt.subplots(dimx, dimy, figsize=(dimy * 3.5, dimx * 3.5), dpi=300)
    from plot_tb_events import make_plots

    for i, env in enumerate(envs_list):
        ax = axarr[i // dimy][i % dimy] if dimx > 1 else axarr[i % dimy]
        title_name = env.split('-')[0] if '-' in env else env
        ax.set_title(title_name)
        ax.set_xlabel("Epoch")
        if i % 4 == 0:
            ax.set_ylabel("Episodic return")
        # ax.set_ylim(-1000, 5000)
        # ax.set_xlim(0, 1000)
        new_keys_mapping = {f'{k}/{env}.': v for k, v in keys_mapping.items()}

        kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping, first_epochs=1460)
        make_plots(ax, env, kw_epochs, kw_eprewmean, tags_to_plot, new_keys_mapping)

    # === Plot and save each scalar tag ===
    output_dir = "tensorboard_plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")

    ax0 = fig.add_subplot(111, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    plt.tight_layout()
    output_dir = "figs"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")
    plt.savefig(filename)
    plt.close()
    print("\nAll plots saved to:", title)

def create_table_muojoc_shared():
    scalar_tag = 'train/eprewmean'
    events_dir = "plotting_logs/0110_shared_fixed_adv/"
    envs_list = ['swimmer', 'hopper', 'halfcheetah', 'walker2d', 'ant', 'humanoid', 'humanoidstandup']
    keys_mapping = {
        f'shared.adv.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.1.lr_0.05': 'RAT',
        f'shared.kfac.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.1.lr_0.001': 'ACKTR',
        f'shared.ppo.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.1.lr_0.001': 'PPO',
        f'shared.diag.karzmarz_False.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.1.lr_0.05': 'DiagFisher',
    }

    float_formatter = "{:.1f}".format
    for i, env in enumerate(envs_list):
        new_keys_mapping = {f'{k}/{env}.': v for k, v in keys_mapping.items()}
        kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping, first_epochs=600)
    
        print('------------------------------')
        for tag in tags_to_plot:
            key = new_keys_mapping[tag]
            values = np.stack(kw_eprewmean[tag], axis=0)
            y_mean = np.mean(values, axis=0)
            y_stderr = np.std(values, axis=0) / np.sqrt(len(values))
            print(env, key, ' Ep_rew mean: ', float_formatter(y_mean[-1]), ' Ep_rew stderr: ', float_formatter(y_stderr[-1]))

def make_ablation():
    # num_iters = [1, 2, 3, 4]
    num_epochs = 500
    env = 'ant'
    title = f'ablation-{env}-{num_epochs}'
    scalar_tag = 'train/eprewmean'
    events_dir = "plotting_logs/ablations_new/"
    envs_list = [env]
    dimx, dimy = (int(np.ceil(len(envs_list)/4)), 4)
    fig, axarr = plt.subplots(dimx, dimy, figsize=(dimy * 1.8, dimx * 3.5), dpi=300, sharey=True)
    from plot_tb_events import make_plots

    ax = axarr[0]
    ax.set_title('(a) RAT and Grad Clip')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Episodic return (K)")
    keys_mapping = {
        f'adv.karzmarz_True.rat_False.norm_clip_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': 'W/o RAT',
        f'adv.karzmarz_True.rat_True.norm_clip_False.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': 'W/o grad clip',
        f'adv.karzmarz_True.rat_True.norm_clip_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': 'Full',
    }
    new_keys_mapping = {f'{k}/{env}.': v for k, v in keys_mapping.items()}
    kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping, first_epochs=num_epochs, divide_1k=True)
    make_plots(ax, env, kw_epochs, kw_eprewmean, tags_to_plot, new_keys_mapping)

    ax = axarr[1]
    ax.set_title('(b) Batch sizes')
    ax.set_xlabel("Batch size")
    # ax.set_ylabel("Episodic return")
    keys_mapping = {
        f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_4.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': '128',
        f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_8.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': '256',
        f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_16.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': '512',
        f'adv.karzmarz_True.rat_True.norm_clip_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': '1024',
    }
    new_keys_mapping = {f'{k}/{env}.': v for k, v in keys_mapping.items()}
    kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping, first_epochs=num_epochs, divide_1k=True)
    batch_size_to_keys = {
        r'$2^7$': f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_4.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05/{env}.', 
        r'$2^8$': f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_8.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05/{env}.', 
        r'$2^9$': f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_16.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05/{env}.', 
        r'$2^{10}$': f'adv.karzmarz_True.rat_True.norm_clip_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05/{env}.', 
    }
    make_bar_plots(ax, env, batch_size_to_keys, kw_epochs, kw_eprewmean, tags_to_plot, new_keys_mapping, color_idx=0)

    ax = axarr[2]
    ax.set_title('(c) Kaczmarz Iterations')
    ax.set_xlabel("Iteration")
    # ax.set_ylabel("Episodic return")
    keys_mapping = {
        f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e1x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': '1x8',
        f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e2x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': '2x8',
        f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e3x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': '3x8',
        f'adv.karzmarz_True.rat_True.norm_clip_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': '4x8',
    }
    new_keys_mapping = {f'{k}/{env}.': v for k, v in keys_mapping.items()}
    kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping, first_epochs=num_epochs, divide_1k=True)
    iter_to_keys = {
        r'$1\times8$': f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e1x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05/{env}.', 
        r'$2\times8$': f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e2x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05/{env}.', 
        r'$3\times8$': f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e3x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05/{env}.', 
        r'$4\times8$': f'adv.karzmarz_True.rat_True.norm_clip_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05/{env}.', 
    }
    # make_plots(ax, env, kw_epochs, kw_eprewmean, tags_to_plot, new_keys_mapping)
    make_bar_plots(ax, env, iter_to_keys, kw_epochs, kw_eprewmean, tags_to_plot, new_keys_mapping, color_idx=1)

    ax = axarr[3]
    ax.set_title(r'(d) Damping $\lambda$')
    ax.set_xlabel(r"Coefficient $\lambda$")
    # ax.set_ylabel("Episodic return")
    keys_mapping = {
        f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.01.lr_pi_0.05': '0.01',
        f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.05.lr_pi_0.05': '0.05',
        f'adv.karzmarz_True.rat_True.norm_clip_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05': '0.1',
        f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.2.lr_pi_0.05': '0.2',
        f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.4.lr_pi_0.05': '0.4',
        f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.8.lr_pi_0.05': '0.8',
    }
    new_keys_mapping = {f'{k}/{env}.': v for k, v in keys_mapping.items()}
    kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping, first_epochs=num_epochs, divide_1k=True)
    iter_to_keys = {
        r'$.01$': f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.01.lr_pi_0.05/{env}.', 
        r'$.05$': f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.05.lr_pi_0.05/{env}.', 
        r'$.1$':  f'adv.karzmarz_True.rat_True.norm_clip_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05/{env}.', 
        r'$.2$':  f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.2.lr_pi_0.05/{env}.', 
        r'$.4$':  f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.4.lr_pi_0.05/{env}.', 
        r'$.8$':  f'adv.karzmarz_True.rat_True.norm_clip_True.num_envs_32.mlp.a256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.8.lr_pi_0.05/{env}.', 
    }
    # make_plots(ax, env, kw_epochs, kw_eprewmean, tags_to_plot, new_keys_mapping)
    make_bar_plots(ax, env, iter_to_keys, kw_epochs, kw_eprewmean, tags_to_plot, new_keys_mapping, color_idx=2)

    # === Plot and save each scalar tag ===
    output_dir = "tensorboard_plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")

    ax0 = fig.add_subplot(111, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    plt.tight_layout()
    output_dir = "figs"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")
    plt.savefig(filename)
    plt.close()
    print("\nAll plots saved to:", title)

if __name__ == "__main__":
    # make_line_plots_dropout()
    # make_training_plot()
    # make_humanoid_iters()
    # make_training_plot_cnn()
    # make_mujoco_separate()
    # make_mujoco_shared()
    # create_table_muojoc_shared()
    make_ablation()
    # make_procgen_shared()