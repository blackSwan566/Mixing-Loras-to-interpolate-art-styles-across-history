import torch
from diffusers import StableDiffusionPipeline
from lora_diffusion import tune_lora_scale, monkeypatch_add_lora


def inference_lora(config: dict, base_dir: str, device: str):
    """
    inference the trained LoRA weights

    :param config: the config data for the inference e.g. path to model weights
    :param base_dir: where to save the generated images
    :param device: whether we train on cpu or gpu
    """
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
