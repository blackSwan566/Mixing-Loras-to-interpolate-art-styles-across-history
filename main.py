import argparse
from src.utils.functions import load_config
from src.scripts.lora_fine_tune import training_lora
from src.scripts.lora_inference import inference_lora
from src.scripts.create_webdatasets import precompute_data
import torch


def main(args):
    config, base_dir = load_config(task=args.task)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.task == 'training':
        training_lora(config, base_dir, device)

    elif args.task == 'inference':
        inference_lora(config, base_dir, device)

    elif args.task == 'precompute':
        precompute_data(config, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine Tune diffusion model')

    subparsers = parser.add_subparsers(
        dest='task', required=True, help='run specific task'
    )

    training_parser = subparsers.add_parser(
        'training', help='Fine Tune a given diffusion model with LoRA'
    )

    inference_parser = subparsers.add_parser(
        'inference', help='Inference of the fine-tuned diffusion model'
    )

    precomputed_parser = subparsers.add_parser(
        'precompute', help='precomputed the data'
    )

    args = parser.parse_args()
    main(args)
