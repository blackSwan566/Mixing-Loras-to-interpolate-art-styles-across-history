import torch
from lora_diffusion import (
    monkeypatch_add_lora,
    tune_lora_scale,
    patch_pipe,
    monkeypatch_or_replace_lora,
)
from diffusers import (
    StableDiffusionPipeline,
)


def merge_loras_v2(config: dict, base_dir: str, device: str):
    # Load weights from .pt
    lora1 = torch.load(config['w1'])
    lora2 = torch.load(config['w2'])

    pipe = StableDiffusionPipeline.from_pretrained(
        config['general_model_name'],
        torch_dtype=torch.float16,
    ).to(device)
    pipe.safety_checker = None

    prompt = config['prompt']

    torch.manual_seed(config['seed'])

    image = pipe(
        prompt,
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
    ).images[0]
    image.save(f'{base_dir}/no_lora_{prompt}.png')

    monkeypatch_or_replace_lora(pipe.unet, lora1)

    tune_lora_scale(pipe.unet, 0.7)

    torch.manual_seed(config['seed'])

    image = pipe(
        prompt,
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
    ).images[0]
    image.save(f'{base_dir}/first_lora.png')

    monkeypatch_add_lora(pipe.unet, lora2, alpha=1, beta=1)

    tune_lora_scale(pipe.unet, 0.7)

    torch.manual_seed(config['seed'])

    image = pipe(
        prompt,
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
    ).images[0]
    image.save(f'{base_dir}/second_lora.png')
