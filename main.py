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
from torch.optim import AdamW
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import webdataset as wds
import json
import argparse
import pandas as pd
import re
import os
import io
from copy import deepcopy
import time
import matplotlib.pyplot as plt
import pathlib

def training(config: dict, base_dir: str, device: str):
    start = time.time()

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
    )

    # transform weights from float16 to float32
    for name, param in unet.named_parameters():
        if 'lora' in name:
            param.data = param.data.to(torch.float32)
            param.requires_grad = True


    # prepare data
    image_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    def tokenization(example):
        #text = f'a painting in the art style of "{config["style"].replace("_", " ")}"'
        text = f'A painting in the style of {config["prompt"]}'
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

    # optimizer
    optimizer = AdamW(
        list(itertools.chain(*unet_lora_params)),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
 
    # loss function
    loss_fn = MSELoss()

    # epochs
    epochs = config['epochs']

    # collect losses
    losses = []
    val_losses = []

    # Model saving & patience
    current_loss = 1000
    best_model_state = None
    best_epoch = 0
    patience = config['patience']

    # scaler
    scaler = torch.amp.GradScaler()

    for epoch in range(1, epochs + 1):
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
            with torch.autocast(device_type='cuda', dtype=torch.float16):
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
                running_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_loss = running_loss / train_samples
        losses.append(epoch_loss)
        print(f"Epoch {epoch}, train loss: {epoch_loss:.4f}")


        # val loop
        if epoch % 5 == 0:
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
                        image.to(dtype=torch.float16)
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
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
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
            print(f'Epoch {epoch}, val ;oss: {total_val_loss:.4f}')

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
        
        else:
            val_losses.append(None)

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
    


        
def precompute(config: dict, base_dir: str, device: str):
    # load models
    text_encoder = CLIPTextModel.from_pretrained(
        config['model_name'],
        subfolder='text_encoder',
        variant='fp16',
        torch_dtype=torch.bfloat16,
    ).to(device)
    text_encoder.requires_grad_(False)

    tokenizer = CLIPTokenizer.from_pretrained(
        config['model_name'],
        subfolder='tokenizer',
        variant='fp16',
        torch_dtype=torch.bfloat16,
    )

    vae = AutoencoderKL.from_pretrained(
        config['model_name'],
        subfolder='vae',
        variant='fp16',
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.requires_grad_(False)

    # create prompt
    prompt = f'A painting in the style of {config["prompt"]}'
    tokenized = tokenizer(prompt, padding='max_length', return_tensors='pt')

    # create encoder_hidden_states and attention_mask for unet model
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]


    # transformation
    image_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

     # data paths
    if not os.path.isdir(f'./data/{config["dataset"]}_precompute'):
        os.mkdir(f'./data/{config["dataset"]}_precompute')

    SHARDSDIR = pathlib.Path("DATA-SHARDS")
    SHARDSDIR.mkdir(exist_ok=True, parents=True)

    dataset_path = './data/wikiart/'
    subsets = ['train', 'val', 'test']
    split_path = './data/wikiart_split'
    
    
    for subset in subsets:
        subset_dir = os.path.join(split_path, subset)
        for art_epoch in os.listdir(subset_dir):
            art_epoch_path = os.path.join(subset_dir, art_epoch)

            with wds.ShardWriter(f"{SHARDSDIR}/{art_epoch}/{subset}/shard-%06d.tar", maxcount = 100) as writer:
        
                for img_name in os.listdir(art_epoch_path):
                    if img_name.lower().endswith('.jpg'):
                        file_path = os.path.join(art_epoch_path, img_name)
                        
                        # transform image
                        image_tensor = image_transforms(file_path).unsqueeze(0).to(device, dtype=torch.bfloat16)
                        
                        # forward pass of vae
                        with torch.no_grad():
                            latents = vae.encode(image_tensor).latent_dist.mean * 0.18215
                        
                        dictionary_lat_ehs_am = {
                            'latents.npy': latents.cpu(),
                            'encoder_hidden_states.pth': encoder_hidden_states.cpu(),
                            'attention_mask.pth': attention_mask.cpu()
                        }
                        writer.write(dictionary_lat_ehs_am)
            writer.close()
                

def main(args):
    config, base_dir = load_config(task=args.task)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.task == 'training':
        training(config, base_dir, device)

    elif args.task == 'inference':
        inference(config, base_dir, device)

    elif args.task == 'data_preparation':
        data_preparation(config, base_dir)

    elif args.task == 'precompute':
        precompute(config, base_dir, device)


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

    precomputed_parser = subparsers.add_parser(
        'precompute', help='precomputed the data'
    )

    args = parser.parse_args()
    main(args)
