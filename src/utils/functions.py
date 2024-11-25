import argparse
import yaml


def load_config() -> dict:
    """

    Returns:
        Returns a dictionary with parameters
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--style', type=str, required=True)
    parser.add_argument('--version', type=float, default=1.0)

    arguments = parser.parse_args()

    with open(f'src/data/{arguments.style}/{arguments.version}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config
