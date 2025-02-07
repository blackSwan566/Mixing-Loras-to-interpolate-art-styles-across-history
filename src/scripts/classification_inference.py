import torch
from transformers import ViTForImageClassification
from torchvision.models import resnet50
import torch.nn as nn
import pickle
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from ..utils.functions import extract_image_data
import torch.nn.functional as F
import json


def inference_style_classification(config: dict, base_dir: str, device: str):
    """
    takes a trained model and lets the trained classifier predict the merged images from lora_merge_v1 or lora_merge_v2
    saves a json file with each image and the logits

    :param config: the config data for inference e.g. path to model weights, image paths
    :param base_dir: where the config and the json is saved
    :param device: whether we train on cpu or gpu
    """
    # load labels
    with open(
        f'./data/{config["dataset"]}_classification/label_encoder.pkl', 'rb'
    ) as f:
        label_encoder = pickle.load(f)

    num_classes = len(label_encoder.classes_)
    print(label_encoder.classes_)

    # load model
    if config['model'] == 'vit':
        model = ViTForImageClassification.from_pretrained(
            config['model_weights'], num_labels=num_classes
        )
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
    extracted_data = extract_image_data(config['folder_path'])

    image_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['mean'], std=config['std'],
            ),
        ]
    )

    all_results = {}

    for epoch, prompts in extracted_data.items():
        all_results[epoch] = {}

        for prompt, image_list in prompts.items():
            all_results[epoch][prompt] = {}

            for image_data in image_list:
                image_path = image_data['file_path']
                image = Image.open(image_path).convert('RGB')

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

                    probabilties = F.softmax(logits, dim=-1).cpu().numpy().tolist()

                    # convert probablities to class labels
                    class_probabilities = {}
                    for i, prob in enumerate(probabilties[0]):
                        class_name = label_encoder.inverse_transform([i])[0]
                        class_probabilities[class_name] = float(f'{prob:.4f}')

                    all_results[epoch][prompt][image_path] = {
                        'alpha1': image_data['alpha1'],
                        'alpha2': image_data['alpha2'],
                        'alpha3': image_data['alpha3'],
                        'predicted_probabilities': class_probabilities,
                        'true_label': epoch,
                    }

    output_file = f'{base_dir}/results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f'Results saved to {output_file}')
