import time
from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from lora_diffusion import (
    inject_trainable_lora,
    save_lora_weight,
)
import glob
import webdataset as wds
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import itertools
import io


def training_lora(config: dict, base_dir: str, device: str):
    """
    fine tunes a given model with LoRA

    :param config: the config data for the training e.g. model
    :param base_dir: where to save the weights
    :param device: whether we train on cpu or gpu
    """
    start = time.time()

    # load models
    scheduler = DDPMScheduler.from_pretrained(
        config['model_name'],
        subfolder='scheduler',
        variant='fp16',
        torch_dtype=torch.float16,
    )

    unet = UNet2DConditionModel.from_pretrained(
        config['model_name'],
        subfolder='unet',
        variant='fp16',
        torch_dtype=torch.float16,
    ).to(device)
    unet.requires_grad_(False)

    text_encoder = CLIPTextModel.from_pretrained(
        config['model_name'],
        subfolder='text_encoder',
        variant='fp16',
        torch_dtype=torch.float16,
    ).to(device)
    text_encoder.requires_grad_(False)

    tokenizer = CLIPTokenizer.from_pretrained(
        config['model_name'],
        subfolder='tokenizer',
        variant='fp16',
        torch_dtype=torch.float16,
    )

    # create attention_mask and encoder hidden states
    text = f'A painting in the style of {config["prompt"]}'
    tokenized = tokenizer(text, padding='max_length', return_tensors='pt')

    attention_masks = tokenized['attention_mask']
    one_attention_mask = torch.stack(
        [torch.tensor(mask) for mask in attention_masks]
    ).to('cpu')

    input_ids = tokenized['input_ids'].to(device)
    input_ids = torch.stack([torch.tensor(ids) for ids in input_ids])
    one_encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0].to('cpu')

    # free space
    del tokenizer
    del text_encoder
    del input_ids
    torch.cuda.empty_cache()

    # inject LoRA
    unet_lora_params, _ = inject_trainable_lora(
        model=unet,
        r=config['r'],
    )

    # transform weights from float16 to float32
    for name, param in unet.named_parameters():
        if 'lora' in name:
            param.data = param.data.to(torch.float32)
            param.requires_grad = True

    # map numpy to tensor
    def numpy_to_tensor(sample, key):
        data = sample[key]
        numpy_data = np.load(io.BytesIO(data))
        tensor_data = torch.from_numpy(numpy_data).to(dtype=torch.float16).squeeze()

        return tensor_data

    def map_data(sample):
        latents = numpy_to_tensor(sample, 'latents.npy')
        name = sample['__key__']

        return (name, latents)

    # train data
    train_path = glob.glob(
        f'./data/{config["dataset"]}_precomputed/train/{config["style"]}/data-*.tar'
    )
    train_dataset = (
        wds.WebDataset(train_path)
        .shuffle(1024)
        .map(map_data)
        .batched(config['batch_size'])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=False)

    # val data
    val_path = glob.glob(
        f'./data/{config["dataset"]}_precomputed/val/{config["style"]}/data-*.tar'
    )
    val_dataset = (
        wds.WebDataset(val_path)
        .shuffle(1024)
        .map(map_data)
        .batched(config['batch_size'])
    )
    val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=False)

    # optimizer
    optimizer = AdamW(
        list(itertools.chain(*unet_lora_params)),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    # loss function
    loss_fn = MSELoss()

    # epochs
    epochs = config['epochs']

    # collect losses
    losses = []
    val_losses = []

    # Model saving & patience
    current_loss = 1000
    best_model_state = None
    best_epoch = 0
    patience = config['patience']

    # scaler
    scaler = torch.amp.GradScaler()

    # counter for dataset lengs
    train_counter = 0
    val_counter = 0

    for epoch in range(1, epochs + 1):
        unet.train()
        running_loss = 0.0

        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f'Epoch:[{epoch}|{epochs}]')
        ):
            optimizer.zero_grad(set_to_none=True)

            # load data to device
            _, latents = batch
            latents = latents.to(device)

            # duplicate encoder_hidden_states and attention_mask to batch size
            encoder_hidden_states = one_encoder_hidden_states.repeat(
                latents.shape[0], 1, 1
            ).to(device)
            attention_mask = one_attention_mask.repeat(latents.shape[0], 1).to(device)

            # create noisy latents
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            )

            timesteps = timesteps.long()

            noise = torch.randn_like(latents)
            noise_latents = scheduler.add_noise(latents, noise, timesteps)

            # forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = unet(
                    sample=noise_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                ).sample

                # get target value
                if scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif scheduler.config.prediction_type == 'v_prediction':
                    target = scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f'Unknown prediction type {scheduler.config.prediction_type}'
                    )

                # loss calculation
                loss = loss_fn(output.float(), target.float())
                running_loss += loss.item()

            train_counter += len(latents)

            # free space
            del latents
            del timesteps
            del noise
            del noise_latents
            del encoder_hidden_states
            del attention_mask
            del output
            del target
            torch.cuda.empty_cache()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_loss = running_loss / train_counter
        losses.append(epoch_loss)
        print(f'Epoch {epoch}, train loss: {epoch_loss:.4f}')

        # val loop

        unet.eval()
        val_loss = 0.0

        with torch.no_grad():
            for _, batch in enumerate(tqdm(val_dataloader)):
                # load data to device
                _, latents = batch
                latents = latents.to(device)

                # duplicate encoder_hidden_states and attention_mask to batch size
                encoder_hidden_states = one_encoder_hidden_states.repeat(
                    latents.shape[0], 1, 1
                ).to(device)
                attention_mask = one_attention_mask.repeat(latents.shape[0], 1).to(
                    device
                )

                # create noise latents
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                )
                timesteps = timesteps.long()

                noise = torch.randn_like(latents)
                noise_latents = scheduler.add_noise(latents, noise, timesteps)

                # forward pass
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = unet(
                        sample=noise_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=attention_mask,
                    ).sample

                    # get target value
                    if scheduler.config.prediction_type == 'epsilon':
                        target = noise
                    elif scheduler.config.prediction_type == 'v_prediction':
                        target = scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f'Unknown prediction type {scheduler.config.prediction_type}'
                        )

                    # loss calculation
                    loss = loss_fn(output.float(), target.float())
                    val_loss += loss.item()

                val_counter += len(latents)

                # free space
                del latents
                del timesteps
                del noise
                del noise_latents
                del encoder_hidden_states
                del attention_mask
                del output
                del target
                torch.cuda.empty_cache()

        total_val_loss = val_loss / val_counter
        val_losses.append(total_val_loss)
        print(f'Epoch {epoch}, val loss: {total_val_loss:.4f}')

        if current_loss > total_val_loss:
            current_loss = total_val_loss

            # save best model state
            unet.to('cpu')
            best_model_state = deepcopy(unet)
            unet.to(device)

            best_epoch = epoch
            patience = config['patience']

        else:
            patience -= 1

        if patience == 0:
            break

    print(f'best epoch {best_epoch}')
    print('LoRA training finished')
    print(f'train losses: {losses}')
    print(f'val losses: {val_losses}')
    save_lora_weight(best_model_state, f'{base_dir}/{config["style"]}_lora_weight.pt')

    # plot losses
    range_epochs = range(1, epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(range_epochs, losses, '-b', label='Training Loss')
    plt.plot(range_epochs, val_losses, '-r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'{base_dir}/losses.png')

    print(f'it took {time.time() - start} seconds')
