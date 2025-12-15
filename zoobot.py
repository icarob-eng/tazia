import timm
import torch
import torchvision as tv
from PIL import Image

encoder = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_nano', pretrained=True, num_classes=0)
encoder.eval()

preprocess = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Resize(encoder.default_cfg['input_size'][1:]),
    tv.transforms.Normalize(
        mean=encoder.default_cfg['mean'], std=encoder.default_cfg['std']
    )
])


print('Encoder input:', encoder.default_cfg['input_size'])
print('Encoder output:', encoder.num_features)
print('Classifier:', encoder.get_classifier())


input_image = Image.open('data/images_gz2/images/11.jpg').convert('RGB')

input_image = preprocess(input_image).unsqueeze(0)

print('Image shape:', input_image.shape)
print('Image type:', input_image.type())

with torch.no_grad():
    features = encoder(input_image)

print(f'Feature vector shape:', features.shape)

