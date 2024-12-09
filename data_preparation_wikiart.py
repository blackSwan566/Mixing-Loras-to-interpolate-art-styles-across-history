import os
import json
import tarfile
import webdataset as wds
import shutil
import tempfile

dataset = "data/wikiart_split"
subsets = ["test", "train", "val"]  
path_for_split = "data/alldata_wikiart/wikiart"

# pip install split-folders
def split_wikiart():
    splitfolders.ratio(path,seed=1337, output="wikiart_split", ratio=(0.6, 0.2, 0.2))
    
def create_tarfiles():
    #loop train, test val
    for subset in subsets:
        subset_dir = os.path.join(dataset, subset)
        # loop art epochs
        for art_epoch in os.listdir(subset_dir):
            art_epoch_path = os.path.join(subset_dir, art_epoch)
            if os.path.isdir(art_epoch_path):
                tar_name = f"{subset.lower()}_{art_epoch.lower()}.tar"
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    with tarfile.open(tar_name, "w") as tar:
                        for img_name in os.listdir(art_epoch_path):
                            if img_name.lower().endswith('.jpg'):
                                file_path = os.path.join(art_epoch_path, img_name)

                                # Sample Key -> Dateiname ohne Endung
                                sample_key = os.path.splitext(img_name)[0]

                                # JSON-Datei -> metadata
                                json_data = {"art_epoch": art_epoch}

                                json_img_name = f"{sample_key}.json"
                                json_file_path = os.path.join(tmpdir, json_img_name)

                                # write meta data in to json
                                with open(json_file_path, "w", encoding="utf-8") as jf:
                                    json.dump(json_data, jf, ensure_ascii=False)

                                # img & JSON to tar 
                                tar.add(file_path, arcname=img_name)         
                                tar.add(json_file_path, arcname=json_img_name)  

                print(f"Erstellt: {tar_name}")

def create_webdataset():
   
   # Afterwards I changed the path because I added a tar folder in data folder, check out that path is correct at yourse

    dataset = (
        wds.WebDataset('data/tars/train_expressionism.tar').decode('pil').to_tuple('jpg', 'json')
    )

    # to test if webdataset is stuffed
    for image, metadata in dataset:
        print(metadata)  # JSON
        image.show() 
        
#1. split, 2. create tar, 3. create webdataset      
#split_wikiart()
#create_tarfiles()
create_webdataset()
