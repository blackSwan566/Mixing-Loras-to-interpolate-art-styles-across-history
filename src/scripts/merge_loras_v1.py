import torch
from lora_diffusion import (
    tune_lora_scale,
    monkeypatch_or_replace_lora,
)
from diffusers import (
    StableDiffusionPipeline,
)


# Linear
def merge_loras_v1(config: dict, base_dir: str, device: str):
    # Load weights from .pt
    lora1 = torch.load(config['w1'])
    lora2 = torch.load(config['w2'])

    # assign key to lora weights
    wd1 = {
        f"{'up' if i % 2==0 else 'down'}_{i}": tensor for i, tensor in enumerate(lora1)
    }
    wd2 = {
        f"{'up' if i % 2 == 0 else 'down'}_{i}": tensor
        for i, tensor in enumerate(lora2)
    }

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

    # convert dict to lilst
    list_lora = list(merged_lora.values())

    # load pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        config['general_model_name'],
        torch_dtype=torch.float16,
    ).to(device)
    pipe.safety_checker = None

    prompt = config['prompt']

    # synthesize without lora weights
    torch.manual_seed(config['seed'])
    image = pipe(
        prompt,
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
    ).images[0]
    image.save(f'{base_dir}/no_lora_{prompt}.png')

    # add fused lora
    monkeypatch_or_replace_lora(pipe.unet, list_lora, r=config['r'])
    
    # influence of lora on model
    tune_lora_scale(pipe.unet, config['tune_scale'])

    # synthesize with merged lora weights
    torch.manual_seed(config['seed'])
    image = pipe(
        prompt,
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
    ).images[0]
    image.save(
        f'{base_dir}/{config["blending_alpha1"]}_{config["blending_alpha2"]}_{prompt}.png'
    )
