import timm
import torch
import torchvision as tv
from PIL import Image
import numpy as np
import pandas as pd
import tqdm

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
    head = 5000  # max_n  # select first `head` elements
    feature_vector = np.empty([head, 1 + encoder.num_features])
    feature_vector[:, 0] = assets[:head]

    del assets
    for i in tqdm.tqdm(range(head)):
        try:
            feature_vector[i, 1:] = get_features(f'data/images_gz2/images/{int(feature_vector[i,0])}.jpg')
        except FileNotFoundError:
            feature_vector[i, 1:] = np.nan

    np.savetxt('data/zoobot-encoder-convnext_nano.feature_vector.csv', feature_vector, fmt='%.4f')
