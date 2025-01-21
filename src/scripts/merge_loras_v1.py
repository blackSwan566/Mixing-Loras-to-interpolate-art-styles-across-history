import torch
from lora_diffusion import monkeypatch_add_lora, tune_lora_scale
from diffusers import (
    StableDiffusionPipeline,
)


# Linear
def merge_loras_v1(config: dict, base_dir: str, device: str):
    # Load weights from .pt
    lora1 = torch.load(config['w1'])
    lora2 = torch.load(config['w2'])

    # list to dictonary
    wd1 = {f"{'up' if i % 2 == 0 else 'down'}_{i // 2}": tensor for i, tensor in enumerate(lora1)}
    wd2 = {f"{'up' if i % 2 == 0 else 'down'}_{i // 2}": tensor for i, tensor in enumerate(lora2)}

    alpha1 = config['blending_alpha1']
    alpha2 = config['blending_alpha2']

    # new merged lora
    merged_lora = {}
    for key in wd1.keys():
        if key in wd2:
            # Linear
            merged_lora[key] = alpha1 * wd1[key] + alpha2 * wd2[key]
        else:
            merged_lora[key] = wd1[key]

    unet_weights = {k: v for k, v in merged_lora.items() if 'unet' in k}

    # load pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        config['general_model_name'],
        torch_dtype=torch.float16,
    ).to(device)

    prompt = config['prompt']

    torch.manual_seed(config['seed'])
    
    image = pipe(
        prompt,
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
    ).images[0]
    image.save(f'{base_dir}/no_lora.png')

    # add lora to pipe
    monkeypatch_add_lora(pipe.unet, unet_weights, alpha=0.7)

    # influence of lora on model
    tune_lora_scale(pipe.unet, 0.7)

    torch.manual_seed(config['seed'])
    image = pipe(
        config['prompt'],
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
    ).images[0]
    image.save(
        f'{base_dir}/{config["adapter_weights_a1"]}_{config["adapter_weights_a2"]}_{prompt}.png'
    )
