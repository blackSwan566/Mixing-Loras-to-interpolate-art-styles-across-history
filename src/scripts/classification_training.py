import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision.models import resnet50
from torchvision import transforms
from torch import nn
import webdataset as wds
import glob
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy


def train_style_classification(config: dict, base_dir: str, device: str):
    # load labels
    with open(
        f'./data/{config["dataset"]}_classification/label_encoder.pkl', 'rb'
    ) as f:
        label_encoder = pickle.load(f)

    if config['model'] == 'vit':
        processor = ViTImageProcessor.from_pretrained(config['model_name'])
        model = ViTForImageClassification.from_pretrained(
            config['model_name'], num_labels=len(label_encoder.classes_)
        ).to(device)

    else:
        model = resnet50(pretrained=True)
        last_dim = model.fc.in_features
        model.fc = nn.Linear(last_dim, len(label_encoder.classes_))
        model = model.to(device)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['mean'], std=config['std'],
            ),
        ]
    )

    def process_train_sample(sample):
        if config['model'] == 'vit':
            sample['image.jpg'] = train_transform(sample['image.jpg'])
            inputs = processor(
                sample['image.jpg'],
                return_tensors='pt',
                do_resize=True,
                size=224,
                do_normalize=True,
                image_mean=[0.5537, 0.4837, 0.4230],
                image_std=[1346.6721, 1176.6603, 1029.2145],
            )

        else:
            image = sample['image.jpg']
            image_tensor = train_transform(transforms.ToTensor()(image))
            normalize = transforms.Normalize(
                mean=[0.5537, 0.4837, 0.4230], std=[1346.6721, 1176.6603, 1029.2145]
            )
            image_tensor = normalize(image_tensor)
            inputs = {'pixel_values': image_tensor.unsqueeze(0)}

        inputs['labels'] = torch.tensor(int(sample['label.txt']), dtype=torch.long)

        return inputs

    def process_val_sample(sample):
        if config['model'] == 'vit':
            inputs = processor(
                sample['image.jpg'],
                return_tensors='pt',
                do_resize=True,
                size=224,
                do_normalize=True,
                image_mean=[0.5537, 0.4837, 0.4230],
                image_std=[1346.6721, 1176.6603, 1029.2145],
            )

        else:
            image = sample['image.jpg']
            image_tensor = val_transform(image)
            inputs = {'pixel_values': image_tensor.unsqueeze(0)}

        inputs['labels'] = torch.tensor(int(sample['label.txt']), dtype=torch.long)

        return inputs

    def collate_fn(batch):
        pixel_values, labels = zip(*batch)

        return {
            'pixel_values': torch.stack([x.squeeze() for x in pixel_values]),
            'labels': torch.tensor([x for x in labels]),
        }

    # train data
    train_path = glob.glob(
        f'./data/{config["dataset"]}_classification/train/data-*.tar'
    )
    train_dataset = (
        wds.WebDataset(train_path, shardshuffle=1024)
        .decode('pil')
        .shuffle(1024)
        .map(process_train_sample)
        .to_tuple('pixel_values', 'labels')
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # val data
    val_path = glob.glob(f'./data/{config["dataset"]}_classification/val/data-*.tar')
    val_dataset = (
        wds.WebDataset(val_path, shardshuffle=1024)
        .decode('pil')
        .shuffle(1024)
        .map(process_val_sample)
        .to_tuple('pixel_values', 'labels')
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
    )

    epochs = config['epochs']
    optimizer = AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )

    # counter
    len_train_dataloader = 0
    len_val_dataloader = 0

    # lists for plotting
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    # early stopping
    current_loss = float('inf')
    best_model_state = None
    best_epoch = 1
    patience = config['patience']

    if config['model'] == 'resnet':
        loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f'Epoch:[{epoch}|{epochs}]')
        ):
            optimizer.zero_grad()

            # load data
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            if config['model'] == 'vit':
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            else:
                outputs = model(pixel_values)
                loss = loss_fn(outputs, labels)
                logits = outputs

            # collect accuracy
            _, preds = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

            # collect loss
            running_loss += loss.item()
            len_train_dataloader += 1

            loss.backward()
            optimizer.step()

        avg_loss = running_loss / len_train_dataloader
        train_losses.append(avg_loss)
        print(f'Epoch: {epoch}, Training Loss: {avg_loss:.4f}')

        avg_accuracy = 100 * train_correct / train_total
        train_acc.append(avg_accuracy)
        print(f'Epoch: {epoch}, Training Acc: {avg_accuracy:.4f}')

        # val loop
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0

        with torch.no_grad():
            for _, batch in enumerate(
                tqdm(val_dataloader, desc=f'Epoch:[{epoch}|{epochs}]')
            ):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)

                # forward pass
                if config['model'] == 'vit':
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                else:
                    outputs = model(pixel_values)
                    loss = loss_fn(outputs, labels)
                    logits = outputs

                # collect accuracy
                _, preds = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                # collect loss
                val_loss += loss.item()
                len_val_dataloader += 1

        avg_val_loss = val_loss / len_val_dataloader
        val_losses.append(avg_val_loss)
        print(f'Epoch: {epoch}, Validation Loss: {avg_val_loss:.4f}')

        avg_val_accuracy = 100 * val_correct / val_total
        val_acc.append(avg_val_accuracy)
        print(f'Epoch: {epoch}, Validation Acc: {avg_val_accuracy:.4f}')

        # early stopping check
        if current_loss > avg_val_loss:
            current_loss = avg_val_loss

            # save best model state
            model.to('cpu')
            best_model_state = deepcopy(model)
            model.to(device)

            best_epoch = epoch
            patience = config['patience']

        else:
            patience -= 1

        if patience == 0:
            break

    # safe model
    if config['model'] == 'vit':
        best_model_state.save_pretrained(f'{base_dir}/classification_weights.pth')

    else:
        torch.save(
            best_model_state.state_dict(), f'{base_dir}/classification_weights.pth'
        )

    # create plot
    range_epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(range_epochs, train_losses, '-b', label='Training Loss')
    plt.plot(range_epochs, val_losses, '-r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss; best epoch: {best_epoch}')
    plt.legend()
    plt.grid()
    plt.savefig(f'{base_dir}/losses.png')

    plt.figure(figsize=(10, 5))
    plt.plot(range_epochs, train_acc, '-b', label='Training Accuracy')
    plt.plot(range_epochs, val_acc, '-r', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy; best epoch: {best_epoch}')
    plt.legend()
    plt.grid()
    plt.savefig(f'{base_dir}/accuracy.png')
