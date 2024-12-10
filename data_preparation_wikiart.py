import os
import json
import tarfile
import webdataset as wds
import tempfile
import splitfolders
import io

dataset_path = './data/wikiart/'
subsets = ['train', 'val', 'test']
split_path = './data/wikiart_split'
tar_path = './data/wikiart_tar'


# pip install split-folders
def split_wikiart():
    splitfolders.ratio(
        dataset_path, seed=1337, output=split_path, ratio=(0.6, 0.2, 0.2)
    )


def create_tarfiles():
    # loop train, test val
    for subset in subsets:
        subset_dir = os.path.join(dataset_path, subset)
        # loop art epochs
        for art_epoch in os.listdir(subset_dir):
            art_epoch_path = os.path.join(subset_dir, art_epoch)
            if os.path.isdir(art_epoch_path):
                tar_name = f'{tar_path}/{subset.lower()}_{art_epoch.lower()}.tar'

                with tempfile.TemporaryDirectory() as tmpdir:
                    with tarfile.open(tar_name, 'w') as tar:
                        total_samples = 0

                        for img_name in os.listdir(art_epoch_path):
                            if img_name.lower().endswith('.jpg'):
                                file_path = os.path.join(art_epoch_path, img_name)

                                # Sample Key -> Dateiname ohne Endung
                                sample_key = os.path.splitext(img_name)[0]

                                # JSON-Datei -> metadata
                                json_data = {'art_epoch': art_epoch}

                                json_img_name = f'{sample_key}.json'
                                json_file_path = os.path.join(tmpdir, json_img_name)

                                # write meta data in to json
                                with open(json_file_path, 'w', encoding='utf-8') as jf:
                                    json.dump(json_data, jf, ensure_ascii=False)

                                # img & JSON to tar
                                tar.add(file_path, arcname=img_name)
                                tar.add(json_file_path, arcname=json_img_name)

                                total_samples += 1

                        total_metadata = {'total_samples': total_samples}
                        total_metadata_json = json.dumps(total_metadata)
                        total_metadata_bytes = total_metadata_json.encode('utf-8')

                        total_metadata_info = tarfile.TarInfo(
                            name='total_metadata.json'
                        )
                        total_metadata_info.size = len(total_metadata_bytes)
                        tar.addfile(
                            total_metadata_info, io.BytesIO(total_metadata_bytes)
                        )

                        print(f'total len {total_samples}')

                print(f'Erstellt: {tar_name}')


def create_webdataset():
    # Afterwards I changed the path because I added a tar folder in data folder, check out that path is correct at yourse

    dataset = (
        wds.WebDataset('data/tars/train_expressionism.tar')
        .decode('pil')
        .to_tuple('jpg', 'json')
    )

    # to test if webdataset is stuffed
    for image, metadata in dataset:
        print(metadata)  # JSON
        image.show()


# 1. split, 2. create tar, 3. create webdataset
split_wikiart()
# create_tarfiles()
# create_webdataset()
