from lora_diffusion import tune_lora_scale, monkeypatch_add_lora
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler


# Linear inspired by cloneifsimo
def merge_loras_v2(config: dict, base_dir: str, device: str):
    state_dict_A = torch.load(config['w1'])
    state_dict_B = torch.load(config['w2'])

    wd1 = {f'tensor_{i}': tensor for i, tensor in enumerate(state_dict_A)}
    wd2 = {f'tensor_{i}': tensor for i, tensor in enumerate(state_dict_B)}

    unet_weights_A = {k: v for k, v in wd1.items() if 'unet' in k}
    text_encoder_weights_A = {k: v for k, v in wd1.items() if 'text_encoder' in k}

    unet_weights_B = {k: v for k, v in wd2.items() if 'unet' in k}
    text_encoder_weights_B = {k: v for k, v in wd2.items() if 'text_encoder' in k}

    pipe = StableDiffusionPipeline.from_pretrained(
        config['general_model_name'],
        torch_dtype=torch.float16,
    ).to(device)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    prompt = 'A womand cries in the snow, expressionism, renaissance.'
    image = pipe(
        prompt,
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
    ).images[0]
    image.save(f'{base_dir}/no_lora.png')

    monkeypatch_add_lora(pipe.unet, unet_weights_A)
    monkeypatch_add_lora(
        pipe.text_encoder,
        text_encoder_weights_A,
        target_replace_module=['CLIPAttention'],
    )

    torch.manual_seed(0)

    tune_lora_scale(pipe.unet, 0.7)
    tune_lora_scale(pipe.text_encoder, 0.7)

    monkeypatch_add_lora(pipe.unet, unet_weights_B, alpha=1.0, beta=1.0)
    tune_lora_scale(pipe.unet, 0.4)
    tune_lora_scale(pipe.text_encoder, 0.0)

    image = pipe(
        config['prompt'],
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
    ).images[0]
    image.save(f'{base_dir}/lora.png')
