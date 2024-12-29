import itertools
from src.utils.functions import load_config
import torch
from lora_diffusion import (
    inject_trainable_lora,
    save_lora_weight,
    patch_pipe,
    tune_lora_scale,
)
from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer
from torch.nn import MSELoss
from torch.optim import AdamW
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import webdataset as wds
import argparse
import os
from copy import deepcopy
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io
import glob


def merge_loras():

    pipe = StableDiffusionPipeline.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float32,
        ).to(device)
    pipe.safety_checker = None
        
    pipe.load_lora_weights("lordjia/by-feng-zikai", weight_name="fengzikai_v1.0_XL.safetensors", adapter_name=config['model_name_A'])
    pipe.load_lora_weights("lordjia/by-feng-zikai", weight_name="fengzikai_v1.0_XL.safetensors", adapter_name=config['model_name_B'])
    pipe.set_adapters([config['model_name_A'], config['model_name_B']], )

    generator = torch.manual_seed(0)
    prompt = "A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai"
    image = pipe(prompt, generator=generator, cross_attention_kwargs={"scale": 1.0}).images[0]
    image