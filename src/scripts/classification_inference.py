import torch
from transformers import ViTForImageClassification
from torchvision.models import resnet50
import torch.nn as nn
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image

def inference_style_classification(config: dict, base_dir: str, device: str):
    # load labels
    with open(
        f'./data/{config["dataset"]}_classification/label_encoder.pkl', 'rb'
    ) as f:
        label_encoder = pickle.load(f)

    num_classes = len(label_encoder.classes_)
    print(label_encoder.classes_)

    # load model
    if config['model'] == 'vit':
        model = ViTForImageClassification.from_pretrained(config['model_weights'], num_labels=num_classes)
        model.eval()
        model = model.to(device)

    else:
        model = resnet50(pretrained=False)
        last_dim = model.fc.in_features
        model.fc = nn.Linear(last_dim, len(label_encoder.classes_))
        model.load_state_dict(torch.load(config['model_weights']))
        model.eval()
        model = model.to(device)

    # collect image paths
    folder_path = config['folder_path']
    images_path = [img for img in os.listdir(folder_path)]

    image_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5537, 0.4837, 0.4230], std=[1346.6721, 1176.6603, 1029.2145])
        ]
    )


    predicted_labels = []
    true_labels = []
    images = []

    for image_path in images_path:
        # extract class TODO
        true_labels.append('pop_art')

        image_path = './images/' + image_path
        image = Image.open(image_path).convert('RGB')
        images.append(image)

        pixel_values = image_transforms(image)
        pixel_values = pixel_values.unsqueeze(0)
        pixel_values = pixel_values.to(device)

        # predict class
        with torch.no_grad():
            outputs = model(pixel_values)

            if config['model'] == 'vit':
                logits = outputs.logits

            else:
                logits = outputs

            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            print(f'Image {image_path} predicted class id: {predicted_class_idx}')
            predicted_labels.append(predicted_class_idx)

    # convert labels back to text
    print(predicted_labels)
    predicted_labels = label_encoder.inverse_transform(predicted_labels)
    print(predicted_labels)
    # confusion matrix plot
    # cm = confusion_matrix(predicted_labels, true_labels)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title('confusion matrix')
    # plt.xlabel('predicted labels')
    # plt.ylabel('true label')
    # plt.xticks(rotation=45, ha="right")
    # plt.tight_layout()
    # plt.savefig(f'{base_dir}/confusion_matrix.png')

    # plot images
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))

    for i, image in enumerate(images):
        predicted_label = predicted_labels[i]
        true_label = true_labels[i]

        axes[i].imshow(image)
        axes[i].set_title(f"Predicted: {predicted_label}\nTrue: {true_label}")
        axes[i].axis('off') # Turn off axis labels
    
    plt.tight_layout()
    plt.savefig(f'{base_dir}/images.png')
