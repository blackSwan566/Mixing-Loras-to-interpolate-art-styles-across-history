import itertools
import tarfile
from src.utils.functions import load_config
import torch
from lora_diffusion import (
    inject_trainable_lora,
    save_lora_weight,
)
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler
from torch.nn import MSELoss
from bitsandbytes.optim import AdamW8bit
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import webdataset as wds
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    unet_lora_params, _ = inject_trainable_lora(
        model=unet,
        r=config['r'],
        dropout_p=config['dropout'],
    )

    optimizer = AdamW8bit(
        list(itertools.chain(*unet_lora_params)),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    loss_fn = MSELoss()

    image_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    def tokenization(example):
        text = f'A painting from {example["json"]["author_name"]} with the name {example["json"]["painting_name"]}'
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

    # prepare dataloader
    dataset = (
        wds.WebDataset(f'{config["style"]}_dataset.tar', shardshuffle=1024)
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

            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

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

            encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

            output = unet(
                sample=noise_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
            ).sample

            if scheduler.config.prediction_type == 'epsilon':
                target = noise
            elif scheduler.config.prediction_type == 'v_prediction':
                target = scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f'Unknown prediction type {scheduler.config.prediction_type}'
                )

            loss = loss_fn(output.float(), target.float())
            loss = loss / accumulation_steps
            running_loss += loss.item()

            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == total_samples:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

        epoch_loss = running_loss / total_samples
        losses.append(epoch_loss)
        print(epoch_loss)

    print('LoRA training finished')
    print(losses)
    save_lora_weight(
        unet, f'./src/data/{config["style"]}/{config["version"]}/lora_weight.pt'
    )


if __name__ == '__main__':
    cfg = load_config()
    main(cfg)
