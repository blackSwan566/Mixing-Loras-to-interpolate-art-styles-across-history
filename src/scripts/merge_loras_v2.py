import torch
from lora_diffusion import monkeypatch_or_replace_lora, tune_lora_scale
from diffusers import (
    StableDiffusionPipeline,
)


def merge_loras_v3(config: dict, base_dir: str, device: str):
    # Load weights from .pt
    lora1 = torch.load(config['w1'])
    lora2 = torch.load(config['w2'])
    lora3 = torch.load(config['w3'])

    # list to dictonary
    wd1 = {f'tensor_{i}': tensor for i, tensor in enumerate(lora1)}
    wd2 = {f'tensor_{i}': tensor for i, tensor in enumerate(lora2)}
    wd3 = {f'tensor_{i}': tensor for i, tensor in enumerate(lora3)}

    alpha1 = config['blending_alpha1']
    alpha2 = config['blending_alpha2']
    alpha3 = config['blending_alpha3']

    # new merged lora
    merged_lora = {}
    for key in wd1.keys():
        if key in wd2 and key in wd3:
            # Linear
            merged_lora[key] = alpha1 * wd1[key] + alpha2 * wd2[key] + alpha3 * wd3[key]
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

    # add lora to pipe
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
        f'{base_dir}/{config["blending_alpha1"]}_{config["blending_alpha2"]}_{config["blending_alpha3"]}_{prompt}.png'
    )
