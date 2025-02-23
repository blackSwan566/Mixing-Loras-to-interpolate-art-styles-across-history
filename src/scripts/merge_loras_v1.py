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
    """
    merge two LoRA weights and synthesize an image

    :param config: the config data for the merging e.g. path to model weights
    :param base_dir: where to save the generated images
    :param device: whether we train on cpu or gpu
    """

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

    if config['full_alpha']:
        steps = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        inversed_steps = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]

        for step, inversed_steps in zip(steps, inversed_steps):
            alpha1 = step
            alpha2 = inversed_steps

            generate_image(config, base_dir, alpha1, alpha2, device, wd1, wd2)

    else:
        alpha1 = config['blending_alpha1']
        alpha2 = config['blending_alpha2']

        generate_image(config, base_dir, alpha1, alpha2, device, wd1, wd2)


def generate_image(config, base_dir, alpha1, alpha2, device, wd1, wd2):
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
    image.save(f'{base_dir}/no_lora.png')

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
    image.save(f'{base_dir}/{alpha1}_{alpha2}.png')
