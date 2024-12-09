import itertools
import tarfile
from src.utils.functions import load_config
import torch
from lora_diffusion import (
    inject_trainable_lora,
    save_lora_weight,
    patch_pipe,
    tune_lora_scale,
)
from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler
from torch.nn import MSELoss
from bitsandbytes.optim import AdamW8bit
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import webdataset as wds
from huggingface_hub import get_token
import json
import argparse
import pandas as pd
from datasets import load_dataset
import re
import os
import io


def training(config: dict, base_dir: str, device: str):
    # load models
    scheduler = DDPMScheduler.from_pretrained(
        config['model_name'],
        subfolder='scheduler',
        variant='fp16',
        torch_dtype=torch.float16,
    )

    unet = UNet2DConditionModel.from_pretrained(
        config['model_name'],
        subfolder='unet',
        variant='fp16',
        torch_dtype=torch.float16,
    ).to(device)
    unet.requires_grad_(False)

    text_encoder = CLIPTextModel.from_pretrained(
        config['model_name'],
        subfolder='text_encoder',
        variant='fp16',
        torch_dtype=torch.float16,
    ).to(device)
    text_encoder.requires_grad_(False)

    tokenizer = CLIPTokenizer.from_pretrained(
        config['model_name'],
        subfolder='tokenizer',
        variant='fp16',
        torch_dtype=torch.float16,
    )

    vae = AutoencoderKL.from_pretrained(
        config['model_name'],
        subfolder='vae',
        variant='fp16',
        torch_dtype=torch.float16,
    ).to(device)
    vae.requires_grad_(False)

    # inject LoRA
    unet_lora_params, _ = inject_trainable_lora(
        model=unet,
        r=config['r'],
        dropout_p=config['dropout'],
    )

    # prepare data
    image_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    def tokenization(example):
        text = f'A painting from the art style "{config["style"]}"'
        return tokenizer(text, padding='max_length')

    def apply_transform(sample):
        if 'jpg' not in sample:
            return None

        sample['image'] = image_transforms(sample['jpg'])

        tokenized = tokenization(sample)
        sample['input_ids'] = tokenized['input_ids']
        sample['attention_mask'] = tokenized['attention_mask']

        return sample

    def collate_fn(batch):
        images, input_ids, attention_masks = zip(*batch)

        return {
            'image': torch.stack(images),
            'input_ids': torch.stack([torch.tensor(ids) for ids in input_ids]),
            'attention_mask': torch.stack(
                [torch.tensor(mask) for mask in attention_masks]
            ),
        }
    
    hf_token = get_token()
    url = "https://huggingface.co/datasets/timm/imagenet-12k-wds/resolve/main/imagenet12k-train-{{0000..1023}}.tar"
    url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'"
    dataset = wds.WebDataset(url).decode()
    
    # prepare dataloader
    dataset = (
        wds.WebDataset(
            f'./data/{config["dataset"]}/{config["style"]}_dataset.tar',
            shardshuffle=1024,
        )
        .decode('pil')
        .shuffle(1024)
        .map(apply_transform)
        .to_tuple('image', 'input_ids', 'attention_mask')
    )
    dataloader = DataLoader(
        dataset, config['batch_size'], shuffle=False, collate_fn=collate_fn
    )

    def read_total_samples_from_tar(tar_filename):
        with tarfile.open(tar_filename, 'r') as tar:
            for member in tar.getmembers():
                if member.name == 'total_metadata.json':
                    f = tar.extractfile(member)
                    total_metadata = json.load(f)

                    return total_metadata.get('total_samples', 0)

    total_samples = read_total_samples_from_tar(f'{config["style"]}_dataset.tar')
    epochs = config['epochs']

    # prepare training parameters
    optimizer = AdamW8bit(
        list(itertools.chain(*unet_lora_params)),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    loss_fn = MSELoss()

    num_training_steps = total_samples * epochs
    warmup_steps = int(num_training_steps * 0.1)

    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    losses = []

    accumulation_steps = 32 // config['batch_size']

    for epoch in range(epochs):
        unet.train()
        running_loss = 0.0

        for step, batch in enumerate(
            tqdm(dataloader, desc=f'Epoch:[{epoch + 1}|{epochs}]')
        ):
            optimizer.zero_grad(
                set_to_none=True
            ) if step % accumulation_steps == 0 else None

            # load data to device
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # create noise latents
            latents = vae.encode(image.to(dtype=torch.float16)).latent_dist.sample()
            latents = latents * 0.18215

            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            )

            timesteps = timesteps.long()

            noise = torch.randn_like(latents)
            noise_latents = scheduler.add_noise(latents, noise, timesteps)

            # encode input
            encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

            # forward pass
            output = unet(
                sample=noise_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
            ).sample

            # get target value
            if scheduler.config.prediction_type == 'epsilon':
                target = noise
            elif scheduler.config.prediction_type == 'v_prediction':
                target = scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f'Unknown prediction type {scheduler.config.prediction_type}'
                )

            # loss calculation
            loss = loss_fn(output.float(), target.float())
            loss = loss / accumulation_steps
            running_loss += loss.item()
            loss.backward()

            # gradient accumulation
            if (step + 1) % accumulation_steps == 0 or (step + 1) == total_samples:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

        epoch_loss = running_loss / total_samples
        losses.append(epoch_loss)
        print(epoch_loss)

    print('LoRA training finished')
    print(losses)
    save_lora_weight(unet, f'{base_dir}/lora_weight.pt')


def inference(config: dict, base_dir: str, device: str):
    torch.manual_seed(42)

    # load diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float16,
    ).to(device)
    pipe.safety_checker = None

    # create image without LoRA
    image = pipe(
        config['prompt'],
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
    ).images[0]
    image.save(f'{base_dir}/no_lora.png')

    # load weights
    patch_pipe(
        pipe,
        config['data_path'],
        patch_unet=True,
        patch_text=False,
        patch_ti=False,
    )
    tune_lora_scale(pipe.unet, 1.00)

    # create image with LoRA
    image = pipe(
        config['prompt'],
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
    ).images[0]
    image.save(f'{base_dir}/lora.png')


def data_preparation(config: dict, base_dir: str):
    # declare paths
    image_data = f'./{base_dir}/{config["dataset"]}'
    csv = f'./{base_dir}/{config["dataset"]}_label.csv'
    tar_archive = f'{base_dir}/{config["style"]}_dataset.tar'

    # read csv
    labels = pd.read_csv(csv, sep='\t')
    labels.columns = labels.columns.str.strip()

    # counter
    total_samples = 0

    with tarfile.open(tar_archive, 'w') as tar:
        for _, row in labels.iterrows():
            image_id = row['ID']
            author = row['AUTHOR']
            title = row['TITLE']
            date = row['DATE']

            # Delete everything drom date thats not a number
            date_numbers = re.sub(r'\D', '', date)
            if not date_numbers:
                # skip rows without numbers
                print(f'Skipping row with invalid date: {date}')
                continue

            # filter art epochs by numbers of date: everything <1490 -> middleage everything >= 1490 & <=1600 -> renaissance everything>=1600 & <=1720 -> baroque
            if (
                config['start_date_epoch']
                <= int(date_numbers)
                <= config['end_date_epoch']
            ):
                image_path = os.path.join(image_data, f'{image_id}.jpg')

                if os.path.exists(image_path):
                    tar.add(image_path, arcname=f'{image_id}.jpg')

                    metadata = {
                        'painting_name': title,
                        'author_name': author,
                        'time': row.get('TIMELINE', 'Unknown'),
                        'date': row.get('DATE', 'Unknown'),
                        'location': row.get('LOCATION', 'Unknown'),
                    }
                    json_data = json.dumps(metadata)
                    json_bytes = json_data.encode('utf-8')

                    json_info = tarfile.TarInfo(name=f'{image_id}.json')
                    json_info.size = len(json_bytes)
                    tar.addfile(json_info, io.BytesIO(json_bytes))

                    total_samples += 1

                else:
                    print(f'Image {image_path} not found')

        total_metadata = {'total_samples': total_samples}
        total_metadata_json = json.dumps(total_metadata)
        total_metadata_bytes = total_metadata_json.encode('utf-8')

        total_metadata_info = tarfile.TarInfo(name='total_metadata.json')
        total_metadata_info.size = len(total_metadata_bytes)
        tar.addfile(total_metadata_info, io.BytesIO(total_metadata_bytes))

        print(f'Tar archive created: {tar_archive}')


def main(args):
    config, base_dir = load_config(task=args.task)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.task == 'training':
        training(config, base_dir, device)

    elif args.task == 'inference':
        inference(config, base_dir, device)

    elif args.task == 'data_preparation':
        data_preparation(config, base_dir)


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

    data_preparation_parser = subparsers.add_parser(
        'data_preparation', help='prepares the data'
    )

    args = parser.parse_args()
    main(args)
