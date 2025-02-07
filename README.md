# Mixing-Loras-to-interpolate-art-styles-across-history

# About the project
Low-Rank Adaptation Models (LoRAs) have been introduced to save computational costs and focus on one task without retraining the entire network (Edward J. et. al). In the present project, we trained three LoRAs based on the CompVis/stable-diffusion-v1-4 model and images from three art eras: Early Renaissance, Expressionism, and Pop Art of the Wikiart dataset. We perform linear interpolation to merge the art eras into two versions. In V1, we merge two art eras; in V2, we merge three to look at paintings that merge across history. We did a qualitative analysis of promising results, on the one hand side, from images from the trained LoRAs and, on the other hand, from the merged LoRA Versions. We have identified recurring patterns regarding style and content for specific art eras and interpolation steps. Even in edge case images, we can find features from the respective art eras. In addition, we used the fine-tuned ResNet50 to classify the merged images into our three epochs. Our results showed that the classifier can classify images created by our merged loRAs as long as only two weights are merged.


# Project Setup
Generally every python command is combined with its corresponding config file in the **/config** folder. Make sure to change the configs in case you want to use different data, models or hyperparamteres.

 Every python execution which saves data like weights or new images will have its config saved in **src/data/ + corresponding command  e.g. training**.

 Also the folder where weights are saved contains the run/version, the date in YY-MM-DD format and the commit hash do backtrack code.
## 1. Install Environment (Python 3.8.10)

You can set up the environment using either **Conda** or **venv**.

### Using Conda:
```bash
conda create --name my_env python=3.8.10 --channel conda-forge
conda activate my_env
```

### Using venv:
Make sure that python 3.8.10 is installed
```bash
python3 -m venv venv
source venv/bin/activate 
```

### Install Requirements
```bash
pip install -r requirement.txt
```

## 2. Download Data
Use the following command to download the required data in this case WikiArt (https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md):
```bash
bash download_data.sh  # Replace with actual command
```

## 3. Fine Tune diffusion model
Before you can fine tune a diffusion model with LoRA the project allows to precompute the latents of the images.

Before executing the command below make sure to check the configs in **config/precompute**. This project uses **CompVis/stable-diffusion-v1-4** as diffusion model but feel free to use another model.
```bash
python main.py precompute
```
The command creates a train and validation split of the data and makes a forward pass through the auto-encoder and save these latents as numpy array in a [WebDataset](https://github.com/webdataset/webdataset)

&nbsp;

After this is done, what's left is to fine tune the model to a specific art epoch. Every art epoch should contain another prompt "Style", thus change the configs for every art epoch. The model is trained with this general prompt:
```bash
f'A painting in the style of {config["prompt"]}'
```

After going through the hyperparameters execute to train
```bash
python main.py training
```
&nbsp;

If you'd like to use our pre-trained weights you can download them from us under the following link:
```bash
https://syncandshare.lrz.de/getlink/fi25yJDzdbBT6G69rE7XxG/training.zip
```
Insert the extracted zip file into **src/data/**

&nbsp;

## 4. Inference of the LoRA model
What is important when trying to inference the LoRA model is to copy and paste the path to the trained weights into the config file **inference.yaml**. The corresponding key is **data_path**
```bash
python main.py inference
```
&nbsp;
## 5. Merge LoRA weights
This project offers the possibility to merge two LoRas at a time or three with **merge_loras_v1** and **merge_loras_v2** respectively. 

Always check the **blending_alpha** values which indicates how much each LoRA weights will be used. 

If in **merge_loras_v1.yaml** the key **full_alpha** is set to true, the script will synthesize images in the following alpha steps:
```bash
steps = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
inversed_steps = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]
```

If the configs are set up, use the following command
```bash
python main.py merge_lora_v1
```
or 
```bash
python main.py merge_lora_v2
```

The results are saved in **src/data/merge_lora_v\*/version/epoch/prompt**

&nbsp;

## 6. Train Classifier
Before we can fine tune a classifier, the data will be once again saved into [WebDataset](https://github.com/webdataset/webdataset)

```bash
python main.py prepare_classification
```

Creates a train and val split and also a pickle file where the art epochs are converted to labels. This is done with scikit-learn [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

The following command computes the mean and std of the dataset for normalization:
```bash
python main.py mean_std
```
Insert the computed values into the config for **classification.yaml** and **classification_inference.yaml**

You can choose between the pretrained model **resnet50** or **google/vit-base-patch16-224**. Change the model name in the corresponding config file **classification.yaml** The keys should be self explainable.
```bash
python main.py classification
```

Afterwards, you can use the following command which creates a json file containing the logits of the classifier for each class.
```bash
python main.py classification_inference
```
This will use the trained classifier for the images from the merged LoRAs. Keep in mind that the config needs to be adjusted. Especially the **model_weights** of the classifier and the **folder_path** where the merged_images are.

&nbsp;

## 7. Create plots
**quantitative_analysis.ipynb** creates the plots for the json from **classification_inference**. You have to adjust the cell file_path to the json file properly. Here is sample plot:

<img src="./src/plots/grid_early_renaissance-expressionism_A%20painting%20of%20a%20woman%20in%20the%20city%20in%20Style1%20and%20Style3.png">


@misc{hu2021loralowrankadaptationlarge,
      title={LoRA: Low-Rank Adaptation of Large Language Models}, 
      author={Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
      year={2021},
      eprint={2106.09685},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2106.09685}, 
}

@article{wikiart,
  title={Improved ArtGAN for Conditional Synthesis of Natural Image and Artwork},
  author={Tan, Wei Ren and Chan, Chee Seng and Aguirre, Hernan and Tanaka, Kiyoshi},
  journal={IEEE Transactions on Image Processing},
  volume    = {28},
  number    = {1},
  pages     = {394--409},
  year      = {2019},
  url       = {https://doi.org/10.1109/TIP.2018.2866698},
  doi       = {10.1109/TIP.2018.2866698}
}