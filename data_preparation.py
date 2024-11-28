import os
import tarfile
import pandas as pd
import json
import io
import re
import webdataset as wds
import torchvision.transforms as transforms
from PIL import Image

# paths
image_data = './data/toy_dataset'
csv = './data/toy_dataset_label.csv'
# tars will be created in the method below
tar_archive_middelage = 'middleage_dataset.tar'
tar_archive_renaissance = 'renaissance_dataset.tar'
tar_archive_barock = 'barock_dataset.tar'
labels = pd.read_csv(csv, sep='\t')
labels.columns = labels.columns.str.strip()


def create_tar(tar_filename, start_date_epoch, end_date_epoch):
    total_samples = 0

    with tarfile.open(tar_filename, 'w') as tar:
        for _, row in labels.iterrows():
            image_id = row['ID']
            author = row['AUTHOR']
            title = row['TITLE']
            date = row['DATE']

            # Delete everything drom date thats not a number
            date_numbers = re.sub(r'\D', '', date)
            if not date_numbers:
                # skip rows without numbers
                print(f'Skipping row with invalid date: {date}')
                continue

            # filter art epochs by numbers of date: everything <1490 -> middelage everything >= 1490 & <=1600 -> reanissance everything>=1600 & <=1720 -> Barock
            if start_date_epoch <= int(date_numbers) <= end_date_epoch:
                image_path = os.path.join(image_data, f'{image_id}.jpg')

                if os.path.exists(image_path):
                    tar.add(image_path, arcname=f'{image_id}.jpg')

                    metadata = {
                        'painting_name': title,
                        'author_name': author,
                        'time': row.get('TIMELINE', 'Unknown'),
                        'date': row.get('DATE', 'Unknown'),
                        'location': row.get('LOCATION', 'Unknown'),
                    }
                    json_data = json.dumps(metadata)
                    json_bytes = json_data.encode('utf-8')

                    json_info = tarfile.TarInfo(name=f'{image_id}.json')
                    json_info.size = len(json_bytes)
                    tar.addfile(json_info, io.BytesIO(json_bytes))

                    total_samples += 1
                else:
                    print(f'Image {image_path} not found')

        total_metadata = {'total_samples': total_samples}
        total_metadata_json = json.dumps(total_metadata)
        total_metadata_bytes = total_metadata_json.encode('utf-8')

        total_metadata_info = tarfile.TarInfo(name='total_metadata.json')
        total_metadata_info.size = len(total_metadata_bytes)
        tar.addfile(total_metadata_info, io.BytesIO(total_metadata_bytes))

        print(f'Tar archive created: {tar_filename}')


# Create tar for art epochs
create_tar(tar_archive_middelage, 827, 1490)
create_tar(tar_archive_renaissance, 1490, 1600)
create_tar(tar_archive_barock, 1590, 1720)
