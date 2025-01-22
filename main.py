import argparse
from src.utils.functions import load_config
from src.scripts.lora_fine_tune import training_lora
from src.scripts.lora_inference import inference_lora
from src.scripts.create_webdatasets import precompute_data, prepare_classificaiton
from src.scripts.classification_training import train_style_classification
from src.scripts.classification_inference import inference_style_classification
from src.scripts.dataset_calculation import calculate_mean_and_std
from src.scripts.create_webdatasets import precompute_data
from src.scripts.merge_loras_v1 import merge_loras_v1
from src.scripts.merge_loras_v2 import merge_loras_v2
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

    elif args.task == 'prepare_classification':
        prepare_classificaiton(config)

    elif args.task == 'classification':
        train_style_classification(config, base_dir, device)
    
    elif args.task == 'classification_inference':
        inference_style_classification(config, base_dir, device)

    elif args.task == 'mean_std':
        calculate_mean_and_std(config)
    elif args.task == 'merge_loras_v1':
        merge_loras_v1(config, base_dir, device)

    elif args.task == 'merge_loras_v2':
        merge_loras_v2(config, base_dir, device)


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

    prepare_classification_parser = subparsers.add_parser(
        'prepare_classification', help='create webdataset for image classification'
    )

    classificaiton_parser = subparsers.add_parser(
        'classification', help='fine tunes a ViT for image classificaiton'
    )

    classificaiton_inference_parser = subparsers.add_parser(
        'classification_inference', help='inferece of the fine tuned ViT'
    )

    mean_std_parser = subparsers.add_parser(
        'mean_std', help='calculates the mean and std for a given dataset'
        )
    
    merge_loras_v1_parser = subparsers.add_parser(
        'merge_loras_v1', help='merge two loras v1'
    )

    merge_loras_v2_parser = subparsers.add_parser(
        'merge_loras_v2', help='merge two loras v2'
    )

    args = parser.parse_args()
    main(args)
