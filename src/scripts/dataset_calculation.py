from ..utils.datasets import SingleFolderDatasetForStats
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from tqdm import tqdm


def calculate_mean_and_std(config: dict):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    dataset = SingleFolderDatasetForStats(
        f'./data/{config["dataset"]}_all_images', transform
    )
    dataloader = DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0

    for batch in tqdm(dataloader):
        batch = batch.float()

        mean += batch.mean(dim=[0, 2, 3]) * batch.size(0)
        std += ((batch - mean.view(1, 3, 1, 1)) ** 2).mean(dim=[0, 2, 3]) * batch.size(
            0
        )

        total_pixels += batch.size(0)

    mean /= total_pixels
    std = (std / total_pixels) ** 0.5

    print(f'mean: {mean}')
    print(f'std: {std}')
