import timm
import torch
import torchvision as tv
from PIL import Image
import numpy as np
import pandas as pd
import tqdm

import os

encoder = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_nano', pretrained=True, num_classes=0)
encoder.eval()

preprocess = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Resize(encoder.default_cfg['input_size'][1:]),
    tv.transforms.Normalize(
        mean=encoder.default_cfg['mean'], std=encoder.default_cfg['std']
    )
])

print('Encoder input:', encoder.default_cfg['input_size'])  # [3, 224, 224]
print('Encoder output:', encoder.num_features)              # 640
print('Classifier:', encoder.get_classifier())              # Identity()


def get_features(path):
    input_image = Image.open(path).convert('RGB')
    input_image = preprocess(input_image).unsqueeze(0)

    # print('Image shape:', input_image.shape)
    # print('Image type:', input_image.type())

    with torch.no_grad():
        return encoder(input_image)

    # print(f'Feature vector shape:', features.shape)

assets = pd.array(pd.read_csv('data/gz2_hart16_classes_simple.csv').asset_id)
max_n = assets.shape[0]

if __name__ == '__main__':
    # ------ configs ------
    src_root = 'data/images_gz2/images'
    outfile = 'data/zoobot-encoder-convnext_nano.feature_vector.csv'
    n = 5000  # max_n  # select first n elements
    batch_size = 1000  # partial saving batches


    try:
        os.remove(outfile)
    except FileNotFoundError:
        pass
    feature_matrix = np.empty([n, 1 + encoder.num_features])
    feature_matrix[:, 0] = assets[:n]

    current_batch = 0

    del assets
    for i in tqdm.tqdm(range(n)):
        try:
            feature_matrix[i, 1:] = get_features(f'{src_root}/{int(feature_matrix[i,0])}.jpg')
        except FileNotFoundError:
            feature_matrix[i, 1:] = np.nan

        if i == 0: continue
        if i % batch_size == 0:
            print('\nSaving batch:', current_batch)
            with open(outfile, 'a+') as f:
                np.savetxt(f, feature_matrix[batch_size * current_batch:batch_size * (current_batch + 1)], fmt='%.4f')
                current_batch+=1

    print('\nSaving last batch:', current_batch)
    with open(outfile, 'a+') as f:
        np.savetxt(f, feature_matrix[batch_size * current_batch:batch_size * (current_batch + 1)], fmt='%.4f')


