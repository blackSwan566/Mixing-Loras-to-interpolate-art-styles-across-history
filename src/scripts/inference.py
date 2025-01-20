import torch
from diffusers import StableDiffusionPipeline
from lora_diffusion import patch_pipe, tune_lora_scale
from src.utils.functions import load_config
import os


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_path = './src/data'

    if not os.path.isdir(f'{base_path}/{config["style"]}/{config["version"]}/images'):
        os.mkdir(f'{base_path}/{config["style"]}/{config["version"]}/images')

    pipe = StableDiffusionPipeline.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float16,
    ).to(device)

    prompt = 'a man with yellow hair and a hoodie and brown eyes'
    torch.manual_seed(1)
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]

    image.save(f'{base_path}/{config["style"]}/{config["version"]}/images/no_lora.png')

    patch_pipe(
        pipe,
        f'{base_path}/{config["style"]}/{config["version"]}/lora_weight.pt',
        patch_unet=True,
        patch_text=False,
        patch_ti=False,
    )

    tune_lora_scale(pipe.unet, 1.00)

    torch.manual_seed(1)
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
    image.save(f'{base_path}/{config["style"]}/{config["version"]}/images/lora.jpg')


if __name__ == '__main__':
    cfg = load_config()
    main(cfg)
