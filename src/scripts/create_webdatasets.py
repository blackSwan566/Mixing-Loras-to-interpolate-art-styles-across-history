import torch
from diffusers import AutoencoderKL
from torchvision import transforms
import os
import webdataset as wds
from tqdm import tqdm
from PIL import Image


def precompute_data(config: dict, device: str):
    """
    precomputes images with a variational autoencoder

    :param config: the config data for the training e.g. model
    :param device: whether we train on cpu or gpu
    """

    # load model
    vae = AutoencoderKL.from_pretrained(
        config['model_name'],
        subfolder='vae',
        variant='fp16',
        torch_dtype=torch.float16,
    ).to(device)
    vae.requires_grad_(False)

    # transformation
    image_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # data path
    shards_dir = f'./data/{config["dataset"]}_precomputed'
    if not os.path.isdir(shards_dir):
        os.mkdir(shards_dir)

    # subsets = ['train', 'val', 'test']
    subsets = ['train']
    split_path = f'./data/{config["dataset"]}_split'

    for subset in subsets:
        subset_dir = os.path.join(split_path, subset)

        # create split folder
        precomputed_subset_dir = shards_dir + f'/{subset}'
        if not os.path.isdir(precomputed_subset_dir):
            os.mkdir(precomputed_subset_dir)

        for art_epoch in os.listdir(subset_dir):
            art_epoch_path = os.path.join(subset_dir, art_epoch)

            # create art epoch folder
            precomputed_subset_epoch_dir = (
                precomputed_subset_dir + f'/{art_epoch.lower()}'
            )
            if not os.path.isdir(precomputed_subset_epoch_dir):
                os.mkdir(precomputed_subset_epoch_dir)

            # wds writer for each epoch
            writer = wds.ShardWriter(
                os.path.join(precomputed_subset_epoch_dir, 'data-%06d.tar'),
                maxsize=500 * 1024 * 1024,
                start_shard=0,
            )

            for index, img_name in enumerate(
                tqdm(
                    os.listdir(art_epoch_path), desc=f'Processing images in {art_epoch}'
                )
            ):
                if img_name.lower().endswith('.jpg'):
                    file_path = os.path.join(art_epoch_path, img_name)

                    # open image
                    image = Image.open(file_path).convert(
                        'RGB'
                    )  # Convert to RGB to ensure consistency

                    # transform image
                    image_tensor = (
                        image_transforms(image)
                        .unsqueeze(0)
                        .to(device, dtype=torch.float16)
                    )

                    # forward pass of vae
                    with torch.no_grad():
                        latents = (
                            vae.encode(image_tensor).latent_dist.sample() * 0.18215
                        )

                    writer_dict = {
                        '__key__': f'{index}_{art_epoch}',
                        'latents.npy': latents.detach().cpu().numpy(),
                    }

                    writer.write(writer_dict)

            writer.close()
