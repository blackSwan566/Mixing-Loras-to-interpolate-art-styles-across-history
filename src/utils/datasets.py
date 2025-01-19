from torch.utils.data import Dataset
from PIL import Image
import os

class SingleFolderDatasetForStats(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []

        for filename in os.listdir(data_dir):
            self.image_paths.append(os.path.join(data_dir, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image