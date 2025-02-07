import yaml
from typing import Tuple
import os
import subprocess
import shutil
from datetime import datetime
import re
import matplotlib.pyplot as plt
from PIL import Image
import textwrap
import numpy as np


def load_config(task: str) -> Tuple[dict, str]:
    """
    Looks for the given yaml and loads it

    :param task: the task e.g. training

    :return: a config file with the parameters and the path to the folder
    """

    file = f'./config/{task}.yaml'
    with open(file, 'r') as f:
        config = yaml.safe_load(f)

    if task == 'precompute' or task == 'prepare_classification' or task == 'mean_std':
        dir_path = f'./src/data/{task}'

        return config, dir_path

    elif 'merge' in task:
        dir_path = f'./src/data/{task}/run{config["version"]}/{config["style"]}/{config["prompt"]}'

        # create folder if not exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # copy config into the folder
        shutil.copy(file, dir_path)

        return config, dir_path

    else:
        if 'data_path' in config and config['data_path'] == '':
            raise ValueError(
                f'"data_path" is empty in the config file for task "{task}"'
            )

        dir_path = './src/data/'

        if 'style' in config:
            task_path = dir_path + task + '/' + config['style']

        else:
            task_path = dir_path + task

        if not os.path.exists(task_path):
            os.makedirs(task_path)

        date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        git_hash = (
            subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            .strip()
            .decode('utf-8')
        )

        run_path = f'{task_path}/run{config["version"]}_{date}_{git_hash}'
        os.makedirs(run_path)

        shutil.copy(file, run_path)

        return config, run_path


def extract_image_data(root_dir: str):
    """
    Extracts image data from a directory structure, with prompts grouped under each epoch.
    Sorts images within each prompt by alpha1 value.

    Args:
      root_dir: The root directory containing the folder structure.

    Returns:
      A dictionary where keys are art epochs and values are dictionaries
      with prompts as keys and list of image data dictionaries as values.
    """

    image_data = {}

    for root, _, files in os.walk(root_dir):
        # extract art epoch
        parts = root.split(os.sep)
        epoch = parts[-2]

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                match_no_lora = re.match(
                    r'no_lora\.png', file
                )  # regex for files called no_lora.png
                match_three_alpha = re.match(
                    r'(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)\.png', file
                )  # regex for files with 3 alpha values
                match_two_alpha = re.match(
                    r'(\d+\.\d+)_(\d+\.\d+)\.png', file
                )  # regex for files with 2 alpha values

                # handles no_lora files
                if match_no_lora:
                    alpha1 = None
                    alpha2 = None
                    alpha3 = None
                    no_lora = True

                    # Extract prompt folder name
                    prompt_folder = os.path.basename(root)
                    if '_no_lora' in prompt_folder:
                        prompt_folder = prompt_folder.replace('_no_lora', '')

                    # Initialize epoch if necessary
                    if epoch not in image_data:
                        image_data[epoch] = {}

                    # Initialize prompt if necessary
                    if prompt_folder not in image_data[epoch]:
                        image_data[epoch][prompt_folder] = []

                    image_data[epoch][prompt_folder].append(
                        {
                            'alpha1': alpha1,
                            'alpha2': alpha2,
                            'alpha3': alpha3,
                            'no_lora': no_lora,
                            'file_path': os.path.join(root, file),
                        }
                    )

                # handles files with 3 alpha values
                elif match_three_alpha:
                    alpha1 = float(match_three_alpha.group(1))
                    alpha2 = float(match_three_alpha.group(2))
                    alpha3 = float(match_three_alpha.group(3))
                    no_lora = False

                    # Extract prompt folder name
                    prompt_folder = os.path.basename(root)

                    if '_no_lora' in prompt_folder:
                        prompt_folder = prompt_folder.replace('_no_lora', '')

                    # Initialize epoch if necessary
                    if epoch not in image_data:
                        image_data[epoch] = {}

                    # Initialize prompt if necessary
                    if prompt_folder not in image_data[epoch]:
                        image_data[epoch][prompt_folder] = []

                    image_data[epoch][prompt_folder].append(
                        {
                            'alpha1': alpha1,
                            'alpha2': alpha2,
                            'alpha3': alpha3,
                            'no_lora': no_lora,
                            'file_path': os.path.join(root, file),
                        }
                    )

                # handles 0.0_1.0 files
                elif match_two_alpha:
                    alpha1 = float(match_two_alpha.group(1))
                    alpha2 = float(match_two_alpha.group(2))
                    alpha3 = None
                    no_lora = False  # flag it as a lora image

                    # Extract prompt folder name
                    prompt_folder = os.path.basename(root)

                    if '_no_lora' in prompt_folder:
                        prompt_folder = prompt_folder.replace('_no_lora', '')

                    # Initialize epoch if necessary
                    if epoch not in image_data:
                        image_data[epoch] = {}

                    # Initialize prompt if necessary
                    if prompt_folder not in image_data[epoch]:
                        image_data[epoch][prompt_folder] = []

                    image_data[epoch][prompt_folder].append(
                        {
                            'alpha1': alpha1,
                            'alpha2': alpha2,
                            'alpha3': alpha3,
                            'no_lora': no_lora,
                            'file_path': os.path.join(root, file),
                        }
                    )

    # sort within prompts
    for epoch_data in image_data.values():
        for prompt_data in epoch_data.values():
            prompt_data.sort(
                key=lambda x: x['alpha1'] if x['alpha1'] is not None else -1
            )

    return image_data


def create_grid_plots(results: dict):
    """
    Creates grid plots for each epoch and prompt.

    Args:
      results: The dictionary containing the image data, probabilities and alpha values.
    """
    for epoch, prompts in results.items():
        for prompt, image_data_dict in prompts.items():
            num_images = len(image_data_dict)

            if num_images == 0:
                continue

            # calculate columns
            num_cols = int(np.ceil(np.sqrt(num_images)))
            num_rows = int(np.ceil(num_images / num_cols))

            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=(20, 15), squeeze=False
            )
            axes = axes.flatten()

            for i, (image_path, data) in enumerate(image_data_dict.items()):
                image = Image.open(image_path).convert('RGB')

                ax = axes[i]
                ax.imshow(image)
                ax.axis('off')

                # create subtext
                alpha_text = (
                    f'Alphas: {data["alpha1"]}, {data["alpha2"]}, {data["alpha3"]}'
                )
                prob_text = ', '.join(
                    f'{k}: {v}' for k, v in data['predicted_probabilities'].items()
                )
                wrapped_prob_text = textwrap.fill(
                    prob_text, width=50
                )  # wrap text for readability

                subtext = f'{alpha_text}\n{wrapped_prob_text}'
                ax.text(
                    0.5,
                    -0.1,
                    subtext,
                    ha='center',
                    va='top',
                    transform=ax.transAxes,
                    fontsize=10,
                )

            fig.suptitle(f'{epoch} - {prompt}', fontsize=16)  # title

            # Hide the empty subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f'./src/plots/grid_{epoch}_{prompt}.png', bbox_inches='tight')
            plt.close(fig)
