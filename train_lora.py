import itertools
from src.utils.functions import load_config
import torch
from lora_diffusion import inject_trainable_lora
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset, Image
from torch.nn import MSELoss
from torch.optim import AdamW
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    text_encoder.requires_grad_(True)

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

    unet_lora_params, train_names = inject_trainable_lora(
        model=unet,
        target_replace_module=config['lora_1']['target_replace_module'],
        r=config['lora_1']['r'],
        dropout_p=config['lora_1']['dropout'],
    )

    optimizer = AdamW(
        itertools.chain(*unet_lora_params, text_encoder.parameters()),
        lr=config['lora_1']['lr'],
        weight_decay=config['lora_1']['weight_decay'],
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
        return tokenizer(example['text'], padding='max_length')

    def transform_data(example):
        example['image'] = [image_transforms(image) for image in example['image']]

        return example

    dataset = load_dataset(
        'lambdalabs/naruto-blip-captions', split='train', trust_remote_code=True
    )
    dataset = dataset.map(tokenization)
    dataset.set_format(
        type='torch', columns=['image', 'text', 'input_ids', 'attention_mask']
    )
    dataset = dataset.cast_column('image', Image(mode='RGB'))
    dataset.set_transform(transform_data)

    dataloader = DataLoader(dataset, config['lora_1']['batch_size'], shuffle=True)

    for epoch in range(config['lora_1']['epochs']):
        unet.train()
        text_encoder.train()
        running_loss = 0.0

        for batch in tqdm(dataloader):
            optimizer.zero_grad()

            image = batch['image'].to(device)
            input_ids = torch.stack([torch.tensor(ids) for ids in batch['input_ids']]).to(device)
            input_ids = input_ids.T
            attention_mask = torch.stack([torch.tensor(mask) for mask in batch['attention_mask']]).to(device)
            attention_mask = attention_mask.T

            latents = vae.encode(image.to(dtype=torch.float16)).latent_dist.sample()
            latents = latents * 0.18215

            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
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

            if scheduler.config.prediction_type == "epsilon":
                target = noise
            elif scheduler.config.prediction_type == "v_prediction":
                target = scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {scheduler.config.prediction_type}"
                )
            
            loss = loss_fn(output.float(), target.float())
            print(loss.item())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(running_loss / len(dataloader))
            



if __name__ == '__main__':
    cfg = load_config()
    main(cfg)
