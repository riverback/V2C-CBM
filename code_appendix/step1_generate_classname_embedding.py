import clip
# import alpha_clip
import numpy as np
import os
from PIL import Image
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CUB')
parser.add_argument('--shots', type=int, default=0, help='0 for no shots')
args = parser.parse_args()

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

dataset_name = args.dataset
if os.path.exists(f'codebooks/{dataset_name}/cls_names.npy'):
    cls_names = np.load(f'codebooks/{dataset_name}/cls_names.npy', allow_pickle=True)
else:
    raise ValueError('Please run step0_generate_word_list.py first')

cls_names = cls_names.tolist()

cls_fewshot_images = {}
if args.shots > 0:
    with open(f'codebooks/{dataset_name}/fewshot_images/img_paths_{args.shots}.txt', 'r') as f:
        fewshot_images = f.readlines()
        fewshot_images = [x.strip() for x in fewshot_images]
        fewshot_images = [(image_path, int(label)) for image_path, label in map(lambda x: x.split(), fewshot_images)]
        for image_path, label in fewshot_images:
            if cls_names[label] not in cls_fewshot_images:
                cls_fewshot_images[cls_names[label]] = []
            cls_fewshot_images[cls_names[label]].append(os.path.join('/mnt/nasv2/hhz/LaBo', image_path))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Generate class name embedding
cls_name_embedding = {}
for cls_name in cls_names:
    if args.shots > 0:
        image_path_list = cls_fewshot_images[cls_name]
        image_list = []
        for image_path in image_path_list:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            image_list.append(image)
        image_list = torch.cat(image_list, dim=0)
        with torch.no_grad():
            image_features = model.encode_image(image_list)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            cls_fewshot_images[cls_name] = image_features.cpu().numpy()
    else:
        with torch.no_grad():
            if args.dataset == 'aircraft':
                text = clip.tokenize([template.format(cls_name).replace('.', 'aircraft.') for template in imagenet_templates]).to(device)
            elif args.dataset == 'DTD':
                text = clip.tokenize([template.format(cls_name).replace('.', 'texture.') for template in imagenet_templates]).to(device)
            else:
                text = clip.tokenize([template.format(cls_name) for template in imagenet_templates]).to(device)
            text_features = model.encode_text(text) 
            text_features /= text_features.norm(dim=-1, keepdim=True)
            cls_name_embedding[cls_name] = text_features.cpu().numpy()

os.makedirs(f'codebooks/{dataset_name}', exist_ok=True)

if args.shots > 0:
    np.save(f'codebooks/{dataset_name}/cls_{args.shots}shot_images_embedding.npy', cls_fewshot_images)
else:
    np.save(f'codebooks/{dataset_name}/cls_name_embedding.npy', cls_name_embedding)