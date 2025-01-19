import torch
from diffusers import AutoencoderKL
from torchvision import transforms
import os
import webdataset as wds
from tqdm import tqdm
from PIL import Image
import random
from sklearn.preprocessing import LabelEncoder
import pickle
import shutil


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

    subsets = ['train', 'val', 'test']
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


def prepare_classificaiton(config: dict):
    """
    takes a dataset e.g wikiart and create webdataset shards for classification task
    """

    # collect all images
    dataset_path = f'./data/{config["dataset"]}_relevant'
    random.seed(38)

    images_with_labels = []
    class_counts = {}

    for art_epoch in os.listdir(dataset_path):
        art_epoch_path = os.path.join(dataset_path, art_epoch)

        if os.path.isdir(art_epoch_path):
            label = art_epoch.lower()
            class_counts[label] = 0

            for img in tqdm(
                os.listdir(art_epoch_path), desc=f'collecting images for {art_epoch}'
            ):
                img_path = os.path.join(art_epoch_path, img)
                images_with_labels.append((img_path, art_epoch.lower()))
                class_counts[label] += 1

    # downsample
    smallest_class_size = min(class_counts.values())    
    balanced_images_with_labels = []

    for label, count in class_counts.items():
        if count > smallest_class_size: 
          class_images = [(img_path, l) for img_path, l in images_with_labels if l == label]
          undersampled_images = random.sample(class_images, smallest_class_size)
          balanced_images_with_labels.extend(undersampled_images)

        else:
          class_images = [(img_path, l) for img_path, l in images_with_labels if l == label]
          balanced_images_with_labels.extend(class_images)

    
    copy_path = f'./data/{config["dataset"]}_all_images'
    os.makedirs(copy_path, exist_ok=True)

    for img_path, label in tqdm(balanced_images_with_labels, desc = "Copying Images"):
        img_name = os.path.basename(img_path)
        new_path = os.path.join(copy_path, f'{label}_{img_name}')
        
        shutil.copy2(img_path, new_path)

    # split data
    random.shuffle(balanced_images_with_labels)
    train_size = int(0.75 * len(balanced_images_with_labels))
    val_size = int(0.25 * len(balanced_images_with_labels))

    dataset = {
        'train': balanced_images_with_labels[:train_size],
        'val': balanced_images_with_labels[train_size : train_size + val_size],
    }

    # transformation
    image_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
        ]
    )

    # create webdataset
    shards_dir = f'./data/{config["dataset"]}_classification'
    os.makedirs(shards_dir, exist_ok=True)

    # create labels & save fitted object
    label_encoder = LabelEncoder()
    art_epochs = [
        art_epoch.lower()
        for art_epoch in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, art_epoch))
    ]
    label_encoder.fit(art_epochs)

    with open(shards_dir + '/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    for subset, data in dataset.items():
        webdataset_subset_dir = shards_dir + f'/{subset}'
        os.makedirs(webdataset_subset_dir, exist_ok=True)

        writer = wds.ShardWriter(
            os.path.join(webdataset_subset_dir, 'data-%06d.tar'),
            maxsize=500 * 1024 * 1024,
            start_shard=0,
        )

        for index, batch in enumerate(
            tqdm(data, desc=f'collecting images from {subset} split')
        ):
            img_path, label = batch
            label_encoded = label_encoder.transform([label])[0]

            image = Image.open(img_path).convert('RGB')
            pil_image = image_transforms(image)

            writer_dict = {
                '__key__': str(index),
                'label.txt': str(label_encoded),
                'image.jpg': pil_image,
                'epoch': label,
            }

            writer.write(writer_dict)

        writer.close()


    