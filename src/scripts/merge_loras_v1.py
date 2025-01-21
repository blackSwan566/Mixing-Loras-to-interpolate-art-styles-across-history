import torch
from lora_diffusion import monkeypatch_add_lora, tune_lora_scale, patch_pipe, monkeypatch_or_replace_lora
from diffusers import (
    StableDiffusionPipeline,
)

    # Linear
def merge_loras_v1(config: dict, base_dir: str, device: str):
    # Load weights from .pt
    lora1 = torch.load(config['w1'])
    lora2 = torch.load(config['w2'])

    # or test i//2
    wd1 = {f"{'up' if i % 2==0 else 'down'}_{i}": tensor for i, tensor in enumerate(lora1)}
    wd2 = {f"{'up' if i % 2 == 0 else 'down'}_{i}": tensor for i, tensor in enumerate(lora2)}
    
    #wd1 = {f"{'tensor'}_{i}": tensor for i, tensor in enumerate(lora1)}
    #wd2 = {f"{'tensor'}_{i}": tensor for i, tensor in enumerate(lora2)}


    alpha1 = config['blending_alpha1']
    alpha2 = config['blending_alpha2']

    # new merged lora
    merged_lora = {}
    for key in wd1.keys():
        if key in wd2:
            # Linear
            merged_lora[key] = alpha1 * wd1[key] + alpha2 * wd2[key]
            # merged_lora[key] = (1 - alpha1) * wd1[key] + alpha1 * wd2[key]
        else:
            merged_lora[key] = wd1[key]

    unet_weights = {k: v for k, v in merged_lora.items() if 'unet' in k}
    text_encoder_weights = {k: v for k, v in merged_lora.items() if "text_encoder" in k}
    
    print(merged_lora)
    print('--------')
    print(merged_lora.keys())
    # load pipe
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

    # add lora to pipe
    # monkeypatch_add_lora(pipe.unet, unet_weights, alpha=0.7)
    # monkeypatch_add_lora(pipe.text_encoder, text_encoder_weights, alpha=0.7)
    
    monkeypatch_or_replace_lora(pipe.unet, unet_weights, r=16)
    #monkeypatch_or_replace_lora(pipe.unet, merged_lora)
    
    tune_lora_scale(pipe.unet, 0.7)
    torch.manual_seed(config['seed'])
    image = pipe(
        prompt,
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=config['guidance_scale'],
    ).images[0]    
    image.save(
         f'{base_dir}/{config["blending_alpha1"]}_{config["blending_alpha2"]}_{prompt}.png'
    )
    # monkeypatch_add_lora(pipe.unet, lora2, alpha=1, beta=1)
    # tune_lora_scale(pipe.unet, 0.7)
    # torch.manual_seed(config['seed'])
    # image = pipe(
    #     prompt,
    #     num_inference_steps=config['num_inference_steps'],
    #     guidance_scale=config['guidance_scale'],
    # ).images[0]    
    # image.save(f'{base_dir}/second_lora.png')


    # influence of lora on model
    # tune_lora_scale(pipe.unet, 0.7)
    # tune_lora_scale(pipe.text_encoder, 0.7)

    # torch.manual_seed(config['seed'])
    # image = pipe(
    #     prompt,
    #     num_inference_steps=config['num_inference_steps'],
    #     guidance_scale=config['guidance_scale'],
    # ).images[0]
    # image.save(
    #     f'{base_dir}/{config["blending_alpha1"]}_{config["blending_alpha2"]}_{prompt}.png'
    # )

# pipeline = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
# ).to("cuda")
# pipeline.load_lora_weights(
#     "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
# )
# pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
# pipeline.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])


