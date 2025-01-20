# some code for later in main - mixing apporaches
import torch
import os 
import numpy as np
from torch.nn import functional as F
from collections import OrderedDict
from lora_diffusion import monkeypatch_add_lora, tune_lora_scale
from diffusers import (

    StableDiffusionPipeline,
)


def merge_loras_v3(config: dict, base_dir: str, device: str):
    torch.seed(config['seed'])
    
    # Load weights from .pt
    lora1 = torch.load(config['w1'])
    lora2 = torch.load(config['w2'])
    lora3 = torch.load(config['w3'])
    # list to dictonary
    wd1 = {f"tensor_{i}": tensor for i, tensor in enumerate(lora1)}
    wd2 = {f"tensor_{i}": tensor for i, tensor in enumerate(lora2)}
    wd3 = {f"tensor_{i}": tensor for i, tensor in enumerate(lora3)}
    
    alpha1 = config['blending-alpha1']
    alpha2 = config['blending-alpha2']
    alpha3 = config['blending-alpha3']
    
    #new merged lora
    merged_lora = {}
    for key in wd1.keys():
        if key in wd2 and key in wd3:
            #Linear
            merged_lora[key] = alpha1 * wd1[key] + alpha2 * wd2[key] + alpha3 * wd3[key]
        else:
            merged_lora[key] = wd1[key]
            
    
    unet_weights = {k: v for k, v in merged_lora.items() if "unet" in k}
    text_encoder_weights = {k: v for k, v in merged_lora.items() if "text_encoder" in k}
    
    #load pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        config['general_model_name'],
        torch_dtype=torch.float16,
    ).to(device)
    
    prompt = config['prompt']

    image = pipe(prompt, num_inference_steps=config['num_inference_steps'], guidance_scale=config['guidance_scale']).images[0]
    image.save(f'{base_dir}/no_lora.png')
    
    #add lora to pipe
    monkeypatch_add_lora(pipe.unet, unet_weights, alpha=0.7)
    monkeypatch_add_lora(pipe.text_encoder, text_encoder_weights, alpha=0.7)

    #influence of lora on model
    tune_lora_scale(pipe.unet, 0.7)
    tune_lora_scale(pipe.text_encoder, 0.7)

    image = pipe(config['prompt'], num_inference_steps= config['num_inference_steps'], guidance_scale=config['guidance_scale']).images[0]
    image.save(f'{base_dir}/lora.png')