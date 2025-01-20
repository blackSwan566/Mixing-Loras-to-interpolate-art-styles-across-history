import yaml
from typing import Tuple
import os
import subprocess
import shutil
from datetime import datetime


def load_config(task: str) -> Tuple[dict, str]:
    """
    Looks for the given yaml and loads it

    :param task: the task e.g. training

    :return: a config file with the parameters and the path to the folder
    """

    file = f'./config/{task}.yaml'
    with open(file, 'r') as f:
        config = yaml.safe_load(f)

    if task == 'data_preparation':
        dir_path = f'./data/{config["dataset"]}'

        return config, dir_path

    else:
        if 'data_path' in config and config['data_path'] == '':
            raise ValueError(
                f'"data_path" is empty in the config file for task "{task}"'
            )

        dir_path = './src/data/'

        task_path = dir_path + task + '/' + config['style']

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
