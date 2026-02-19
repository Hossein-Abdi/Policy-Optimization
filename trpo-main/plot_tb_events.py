import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

import seaborn # sets some style parameters automatically

# COLORS = [(57, 106, 177), (218, 124, 48)] 
COLORS = seaborn.color_palette('colorblind')
matplotlib.rcParams.update({'errorbar.capsize': 2})

def load_event_data(scalar_tag, events_dir, keys_mapping, first_epochs=1220):
    # === List all TensorBoard event files ===
    event_files = []
    for dirpath, _, filenames in os.walk(events_dir):
        for filename in filenames:
            if filename.startswith("events.out.tfevents"):
                full_path = os.path.join(dirpath, filename)
                event_files.append(full_path)

    tags_to_plot = sorted(keys_mapping.keys())

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
            print(f"Loading event file: {event_file_path}")
            ea = event_accumulator.EventAccumulator(event_file_path)
            ea.Reload()

            scalar_tags = ea.Tags()['scalars']
            print(f"Available scalar tags: {scalar_tags}")

            if scalar_tag not in scalar_tags:
                print(f"Scalar tag '{scalar_tag}' not found in event file.")
                continue

            # Get the scalar values for the specified tag
            events = ea.Scalars(scalar_tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])

            if len(steps) < first_epochs or len(values) < first_epochs:
                print(f"Warning: Event file '{event_file_path}' has only {len(steps)} epochs, which is less than the required {first_epochs} epochs. Skipping this file.")
                continue
            if tag not in kw_epochs:
                kw_epochs[tag] = []
            kw_epochs[tag].append(steps[:first_epochs])
            if tag not in kw_eprewmean:
                kw_eprewmean[tag] = []
            kw_eprewmean[tag].append(values[:first_epochs])
    
    return kw_epochs, kw_eprewmean, tags_to_plot

def make_plots(ax, title, kw_epochs, kw_eprewmean, tags_to_plot, keys_mapping):
    color_idx = 0

    for key in tags_to_plot:
        try:
            steps = np.stack(kw_epochs[key], axis=0)
        except (ValueError, KeyError):
            # breakpoint()
            break
        values = np.stack(kw_eprewmean[key], axis=0)

        x = np.mean(steps, axis=0)
        y_mean = np.mean(values, axis=0)
        y_stderr = np.std(values, axis=0) / np.sqrt(len(values))

        ax.plot(x, y_mean, label=keys_mapping[key], color=COLORS[color_idx], linestyle='solid', linewidth=1.0, rasterized=True)
        ax.fill_between(x, y_mean - y_stderr, y_mean + y_stderr, color=COLORS[color_idx], alpha=.25, linewidth=0.0, rasterized=True)
        # plt.xlabel("Epoch")
        # plt.ylabel("Episodic return")
        # plt.title(f"{title}")
        color_idx += 1

    # plt.grid(True)
    ax.legend()

def make_line_plots_dropout():
    title = '0612-mlp-logstd'
    scalar_tag = 'train/eprewmean'
    events_dir = "log_backup/0612_mlp_dropout/"
    envs_list = ['invertedpendulum', 'swimmer', 'hopper', 'walker2d', 'halfcheetah']
    dimx, dimy = (int(np.ceil(len(envs_list)/4)), 4)
    fig, axarr = plt.subplots(dimx, dimy, figsize=(dimy * 3.5, dimx * 3.5), dpi=300)

    dropout_list = [0.0, 0.1, 0.2, 0.3]
    for i, env in enumerate(envs_list):
        ax = axarr[i // dimy][i % dimy] if dimx > 1 else axarr[i % dimy]
        ax.grid(True)
        ax.set_title(env)
        ax.set_xlabel("Dropout rate")
        if i == 0:
            ax.set_ylabel("Episodic return")

        eprewmean = {}
        eprewstderr = {}
        for dropout in dropout_list:
            keys_mapping = {f'empirical.mlp.dropout_{dropout}.linear_0.01.damping_0.0': 'Trust-0.01', 
                            f'empirical.mlp.dropout_{dropout}.linear_0.1.damping_0.0': 'Trust-0.1', 
                            f'empirical.mlp.dropout_{dropout}.linear_0.5.damping_0.0':  'Trust-0.5', 
                            f'empirical.mlp.dropout_{dropout}.linear_1.0.damping_0.0':  'Trust-1.0', 
                            }
            new_keys_mapping = {f'{k}/{env}': v for k, v in keys_mapping.items()}
            _, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping, first_epochs=600)

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
            ax.errorbar(dropout_list, eprewmean[key], yerr=[eprewstderr[key], eprewstderr[key]], label=key, color=COLORS[color_idx], fmt='o', linestyle='solid', linewidth=1.0, rasterized=True)
            color_idx += 1

    ax.legend()

    # === Plot and save each scalar tag ===
    output_dir = "tensorboard_plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")

    ax0 = fig.add_subplot(111, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("\nAll plots saved to:", title)

def make_line_plots_interpolation():
    title = 'full-dropout-mlp-interpolation-no-damping'
    scalar_tag = 'train/eprewmean'
    events_dir = "log_backup/0610_empirical_no_damping/"
    envs_list = ['invertedpendulum', 'swimmer', 'hopper', 'walker2d', 'halfcheetah', 'humanoid']
    dimx, dimy = (int(np.ceil(len(envs_list)/4)), 4)
    fig, axarr = plt.subplots(dimx, dimy, figsize=(dimy * 3.5, dimx * 3.5), dpi=300)

    linear_int_list = [0.0, 0.01, 0.1, 0.5, 1.0]
    for i, env in enumerate(envs_list):
        ax = axarr[i // dimy][i % dimy] if dimx > 1 else axarr[i % dimy]
        ax.grid(True)
        title_name = env.split('-')[0] if '-' in env else env
        ax.set_title(title_name)
        ax.set_xlabel("Interpolation value")
        if i == 0:
            ax.set_ylabel("Episodic return")

        eprewmean = {}
        eprewstderr = {}
        for int_val in linear_int_list:
            keys_mapping = {f'empirical.mlp.dropout_0.0.linear_{int_val}.damping_0.0': 'Dropout-0.0', 
                            f'empirical.mlp.dropout_0.2.linear_{int_val}.damping_0.0': 'Dropout-0.2', 
                            f'empirical.mlp.dropout_0.4.linear_{int_val}.damping_0.0': 'Dropout-0.4', 
                            f'empirical.mlp.dropout_0.6.linear_{int_val}.damping_0.0': 'Dropout-0.6', 
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
            ax.errorbar(linear_int_list, eprewmean[key], yerr=[eprewstderr[key], eprewstderr[key]], label=key, color=COLORS[color_idx], fmt='o', linestyle='solid', linewidth=1.0, rasterized=True)
            color_idx += 1

    ax.legend()

    # === Plot and save each scalar tag ===
    output_dir = "tensorboard_plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")

    ax0 = fig.add_subplot(111, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("\nAll plots saved to:", title)

def tb_plot():
    # === Configuration ===
    # for sigma_type in ['vector', 'mu_shared', 'separate', 'linear']:
    # for epoch in [0, 1, 2, 4]:
    # sigma_type = 'separate'
    # for epoch in [1, 2, 3, 4]:
    # for n_epoch in [1]:
    for _ in ['']:
        # n_epoch = 1
        # dropout = 0.4
        # dropout = 0.0
        # norm_type = 'l2'
        title = f'0216-4envs-cg-shift-fisher-0.5'
        # title = f'1128-fisher-grad-{n_epoch}'
        scalar_tag = 'train/eprewmean'
        events_dir = "log_backups/0216_4envs_shift_prec_fisher_0.5/"

        # envs_list = ['invertedpendulum', 'halfcheetah', 'hopper', 'walker2d', 'swimmer', 'humanoid']
        # envs_list = ['halfcheetah', 'hopper', 'walker2d', 'swimmer', 'humanoid']
        # envs_list = ['halfcheetah', 'hopper', 'walker2d', 'humanoid']
        # envs_list = ['humanoid']
        # envs_list = ['ant', 'halfcheetah', 'hopper', 'swimmer', 'humanoid', 'humanoidstandup']
        envs_list = ['ant', 'swimmer', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup']
        # envs_list = ['ant', 'humanoid']
        dimx, dimy = (int(np.ceil(len(envs_list)/4)), 4)
        keys_mapping = {
            # f"empirical.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_l2_clip_0.5.{sigma_type}.damping_0.1.lr_pi_0.05": "Empirical",
            # f"true.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_l2_clip_0.5.{sigma_type}.damping_0.1.lr_pi_0.05": "True",
            # f"true.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05": "Vector",
            # f"true.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.mu_shared.damping_0.1.lr_pi_0.05": "Mu-shared",
            # f"true.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.separate.damping_0.1.lr_pi_0.05": "Separate",
            # f"true.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.linear.damping_0.1.lr_pi_0.05": "Linear",
            # f"empirical.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_fisher_clip_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": "Empirical",
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_fisher_clip_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": "True",
            # f"empirical.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.5": "vector",
            # f"empirical.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_fisher_clip_0.5.mu_shared.damping_0.1.lr_pi_0.5": "mu-shared",
            # f"empirical.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_fisher_clip_0.5.separate.damping_0.1.lr_pi_0.5": "separate",
            # f"empirical.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_fisher_clip_0.5.linear.damping_0.1.lr_pi_0.5": "linear",
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_fisher_clip_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": "1",
            # f"true.mlp.a256x2x0.0e2x1.c256x2x0.0e4x8.npg_fisher_clip_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": "2",
            # f"true.mlp.a256x2x0.0e3x1.c256x2x0.0e4x8.npg_fisher_clip_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": "3",
            # f"true.mlp.a256x2x0.0e4x1.c256x2x0.0e4x8.npg_fisher_clip_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": "4",
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_none_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": '1', 
            # f"true.mlp.a256x2x0.0e2x1.c256x2x0.0e4x8.npg_none_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": '2', 
            # f"true.mlp.a256x2x0.0e3x1.c256x2x0.0e4x8.npg_none_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": '3', 
            # f"true.mlp.a256x2x0.0e4x1.c256x2x0.0e4x8.npg_none_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": '4', 

            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_none_0.5.vector.damping_0.1.lr_pi_0.5": 'vector', 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_none_0.5.mu_shared.damping_0.1.lr_pi_0.5": 'mu_shared', 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_none_0.5.separate.damping_0.1.lr_pi_0.5": 'separate', 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_none_0.5.linear.damping_0.1.lr_pi_0.5": 'linear', 

            # f"empirical.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.linear.damping_0.1.lr_pi_0.5": 'linear', 
            # f"empirical.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.damping_0.1.lr_pi_0.5": 'vector', 
            # f"empirical.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.mu_shared.damping_0.1.lr_pi_0.5": 'mu_shared', 
            # f"empirical.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.separate.damping_0.1.lr_pi_0.5": 'separate', 

            # f"empirical.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.npg_none_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": '1',
            # f"empirical.mlp.a256x2x0.0e2x1.c256x2x0.0e4x8.npg_none_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": '2',
            # f"empirical.mlp.a256x2x0.0e3x1.c256x2x0.0e4x8.npg_none_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": '3',
            # f"empirical.mlp.a256x2x0.0e4x1.c256x2x0.0e4x8.npg_none_0.5.{sigma_type}.damping_0.1.lr_pi_0.5": '4',

            # f"empirical.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.{sigma_type}.damping_0.1.lr_pi_0.5": 'empirical', 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.{sigma_type}.damping_0.1.lr_pi_0.5": 'true', 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_1_damping_0.1.lr_pi_0.5": '1', 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_5_damping_0.1.lr_pi_0.5": '5', 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_10_damping_0.1.lr_pi_0.5": '10', 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_2_damping_0.1.lr_pi_0.5": '02', 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_4_damping_0.1.lr_pi_0.5": '04', 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_8_damping_0.1.lr_pi_0.5": '08', 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_16_damping_0.1.lr_pi_0.5": '16', 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_32_damping_0.1.lr_pi_0.5": '32', 

            # "woodbury.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.linear.cg_1_damping_0.1.lr_pi_0.5": "linear",
            # "woodbury.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_1_damping_0.1.lr_pi_0.5": "vector",
            # "woodbury.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.mu_shared.cg_1_damping_0.1.lr_pi_0.5": "mu_shared",
            # "woodbury.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.separate.cg_1_damping_0.1.lr_pi_0.5": "separate",

            # "true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_0_damping_0.1_init_False.lr_pi_0.5": "steps-0", 
            # "true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_1_damping_0.1_init_False.lr_pi_0.5": "steps-1", 
            # "true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_2_damping_0.1_init_False.lr_pi_0.5": "steps-2", 
            # "true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_4_damping_0.1_init_False.lr_pi_0.5": "steps-4", 

            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_{epoch}_damping_0.1_init_True.lr_pi_0.5": "w/ init", 
            # f"true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_{epoch}_damping_0.1_init_False.lr_pi_0.5": "w/o init", 

            "true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_0_damping_0.1_shifted_True_prec_False.lr_pi_0.5": 'woodbury', 
            "true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_10_damping_0.1_shifted_False_prec_False.lr_pi_0.5": 'fvg', 
            "true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_10_damping_0.1_shifted_True_prec_False.lr_pi_0.5": 'fvg+shift', 
            "true.mlp.a256x2x0.0e1x1.c256x2x0.0e4x8.vector.cg_10_damping_0.1_shifted_False_prec_True.lr_pi_0.5": 'fvg+prec', 
        }

        fig, axarr = plt.subplots(dimx, dimy, figsize=(dimy * 3.5, dimx * 3.5), dpi=300)

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

            kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping, first_epochs=1000)
            make_plots(ax, env, kw_epochs, kw_eprewmean, tags_to_plot, new_keys_mapping)

        # === Plot and save each scalar tag ===
        output_dir = "tensorboard_plots"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{title}.pdf")

        ax0 = fig.add_subplot(111, frame_on=False)
        ax0.set_xticks([])
        ax0.set_yticks([])

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print("\nAll plots saved to:", title)

if __name__ == "__main__":
    # make_line_plots_dropout()
    # make_line_plots_interpolation()
    tb_plot()