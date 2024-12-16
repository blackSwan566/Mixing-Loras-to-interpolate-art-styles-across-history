import itertools
import tarfile
from src.utils.functions import load_config
import torch
from lora_diffusion import (
    inject_trainable_lora,
    save_lora_weight,
    patch_pipe,
    tune_lora_scale,
    monkeypatch_add_lora
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
from torch.optim import AdamW
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
from copy import deepcopy
import time
import matplotlib.pyplot as plt

def training(config: dict, base_dir: str, device: str):
    start = time.time()

    # load models
    scheduler = DDPMScheduler.from_pretrained(
        config['model_name'],
        subfolder='scheduler',
        variant='fp16',
        torch_dtype=torch.float32,
    )

    unet = UNet2DConditionModel.from_pretrained(
        config['model_name'],
        subfolder='unet',
        variant='fp16',
        torch_dtype=torch.float32,
    ).to(device)
    unet.requires_grad_(False)

    text_encoder = CLIPTextModel.from_pretrained(
        config['model_name'],
        subfolder='text_encoder',
        variant='fp16',
        torch_dtype=torch.float32,
    ).to(device)
    text_encoder.requires_grad_(False)
    # text_encoder = torch.compile(text_encoder)

    tokenizer = CLIPTokenizer.from_pretrained(
        config['model_name'],
        subfolder='tokenizer',
        variant='fp16',
        torch_dtype=torch.float32,
    )

    vae = AutoencoderKL.from_pretrained(
        config['model_name'],
        subfolder='vae',
        variant='fp16',
        torch_dtype=torch.float32,
    ).to(device)
    vae.requires_grad_(False)
    # vae = torch.compile(vae)

    # inject LoRA
    unet_lora_params, _ = inject_trainable_lora(
        model=unet,
        r=config['r'],
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
        text = f'a painting in the art style of "{config["style"].replace("_", " ")}"'
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

    # train data
    train_dataset = (
        wds.WebDataset(
            f'./data/{config["dataset"]}_tar/train_{config["style"]}.tar',
            shardshuffle=1024,
        )
        .decode('pil')
        .shuffle(1024)
        .map(apply_transform)
        .to_tuple('image', 'input_ids', 'attention_mask')
    )

    #train_dataset_limited = itertools.islice(train_dataset, 1)
      
    train_dataloader = DataLoader(
        train_dataset, config['batch_size'], shuffle=False, collate_fn=collate_fn
    )

    # val data
    val_dataset = (
        wds.WebDataset(
            f'./data/{config["dataset"]}_tar/val_{config["style"]}.tar',
            shardshuffle=1024,
        )
        .decode('pil')
        .shuffle(1024)
        .map(apply_transform)
        .to_tuple('image', 'input_ids', 'attention_mask')
    )
    val_dataloader = DataLoader(
        val_dataset, config['batch_size'], shuffle=False, collate_fn=collate_fn
    )

    def read_total_samples_from_tar(tar_filename):
        with tarfile.open(tar_filename, 'r') as tar:
            for member in tar.getmembers():
                if member.name == 'total_metadata.json':
                    f = tar.extractfile(member)
                    total_metadata = json.load(f)

                    return total_metadata.get('total_samples', 0)

    # get train and val length
    train_samples = read_total_samples_from_tar(
        f'./data/{config["dataset"]}_tar/train_{config["style"]}.tar'
    )
    val_samples = read_total_samples_from_tar(
        f'./data/{config["dataset"]}_tar/val_{config["style"]}.tar'
    )

    # prepare training parameters
    # optimizer = AdamW8bit(
    #     list(itertools.chain(*unet_lora_params)),
    #     lr=config['lr'],
    #     weight_decay=config['weight_decay'],
    # )
    optimizer = AdamW(
        list(itertools.chain(*unet_lora_params)),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    # loss function
    loss_fn = MSELoss()

    # epochs
    epochs = config['epochs']

    # learning rate scheduler
    num_training_steps = train_samples * epochs
    warmup_steps = int(num_training_steps * 0.1)

    # lr_scheduler = get_scheduler(
    #     name='linear',
    #     optimizer=optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=num_training_steps,
    # )

    # lr_scheduler.step()

    # losses
    losses = []
    val_losses = []

    # gradient accumulation
    accumulation_steps = 32 // config['batch_size']

    # Model saving & patience
    current_loss = 1000
    best_model_state = None
    best_epoch = 0
    patience = config['patience']

    for epoch in range(epochs):
        unet.train()
        running_loss = 0.0

        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f'Epoch:[{epoch + 1}|{epochs}]')
        ):
            optimizer.zero_grad(set_to_none=True)

            # load data to device
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # create noise latents
            latents = vae.encode(image.to(dtype=torch.float32)).latent_dist.sample()
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
            #loss = loss / accumulation_steps
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            # # gradient accumulation
            # if (step + 1) % accumulation_steps == 0 or (step + 1) == train_samples:
            #     print("Checking optimizer step")
            #     param_before = next(iter(unet.parameters())).clone().detach().cpu() # Get the first layer of the model before the step

            #     torch_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            #     print(f'total graidnet norm: {torch_norm}')

            #     param_updates = []
            #     for group in optimizer.param_groups:
            #         for param in group['params']:
            #             if param.grad is not None:
            #                 param_updates.append(param.grad.abs().mean().item())

            #     if len(param_updates) > 0:
            #         avg_param_update = sum(param_updates) / len(param_updates)
            #         print(f'Average parameter update magnitude: {avg_param_update}')

            #     print(f'Learning rate {lr_scheduler.get_last_lr()}')
            #     optimizer.step()
            #     lr_scheduler.step()
            #     optimizer.zero_grad(set_to_none=True)

            #     param_after = next(iter(unet.parameters())).clone().detach().cpu() # Get the first layer of the model after the step
            #     param_diff = torch.abs(param_before - param_after) # Check the absolute difference
                
            #     print(f'Parameter difference : {torch.mean(param_diff)}')

        epoch_loss = running_loss / train_samples
        losses.append(epoch_loss)
        print("Loss", epoch_loss)


        # val loop
        unet.eval()
        val_loss = 0.0

        with torch.no_grad():
            for _, batch in enumerate(tqdm(val_dataloader)):
                # load data to device
                image = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # create noise latents
                latents = vae.encode(
                    image.to(dtype=torch.float32)
                ).latent_dist.sample()
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
                encoder_hidden_states = text_encoder(input_ids, return_dict=False)[
                    0
                ]

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
                val_loss += loss.item()

        total_val_loss = val_loss / val_samples
        val_losses.append(total_val_loss)
        print(f'Epoch {epoch + 1}, val Loss: {total_val_loss:.4f}')

        if current_loss > total_val_loss:
            current_loss = total_val_loss

            # save best model state
            unet.to('cpu')
            best_model_state = deepcopy(unet)
            unet.to(device)

            best_epoch = epoch
            patience = config['patience']

        else:
            patience -= 1

        if patience == 0:
            break

    print(f'best epoch {best_epoch}')
    print('LoRA training finished')
    print(f'train losses: {losses}')
    print(f'val losses: {val_losses}')
    save_lora_weight(best_model_state, f'{base_dir}/{config["style"]}_lora_weight.pt')

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'{base_dir}/losses.png')

    print(f'it took {time.time() - start} seconds')


def inference(config: dict, base_dir: str, device: str):
    torch.manual_seed(42)

    # load diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float32,
    ).to(device)
    pipe.safety_checker = None

    # torch compile
    pipe.unet = torch.compile(pipe.unet)
    pipe.vae = torch.compile(pipe.vae)

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
        r=config['r'],
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
    # monkeypatch_add_lora(pipe.unet, torch.load(config['data_path']))
    # tune_lora_scale(pipe.unet, 1.00)
    # image = pipe(
    #     config['prompt'],
    #     num_inference_steps=config['num_inference_steps'],
    #     guidance_scale=config['guidance_scale'],
    # ).images[0]
    # image.save(f'{base_dir}/lora.png')



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

        print(f'total length: {total_samples}')
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
