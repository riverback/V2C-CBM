import argparse
import clip
import json
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--number_of_images', '-n', type=int, default=500)
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
    'has a {} on it.',
    'is made of {}.',
    'is a {}.',
    'has many {} on it.',
    'is a type of {}.',
    'part of a {}.',
]


dataset_name = args.dataset

cls_names = np.load(f'codebooks/{dataset_name}/cls_names.npy', allow_pickle=True)
cls_names = cls_names.tolist()
if args.shots > 0:
    imagenet_split_root = f'codebooks/{dataset_name}/imagenet_{args.shots}shot_split'
else:
    imagenet_split_root = f'codebooks/{dataset_name}/imagenet_split'
    
cls_image_ids = {}
for cls_name in cls_names:
    if '/' in cls_name:
        save_cls_name = cls_name.replace('/', '_')
    else:
        save_cls_name = cls_name
    with open(f'{imagenet_split_root}/{save_cls_name}.txt', 'r') as f:
        image_ids = f.readlines()
        image_ids = [x.strip() for x in image_ids][:args.number_of_images]
        cls_image_ids[cls_name] = image_ids

# check the overlap between different classes
for cls_name in cls_names:
    image_ids = cls_image_ids[cls_name]
    for other_cls_name in cls_names:
        if cls_name == other_cls_name:
            continue
        other_image_ids = cls_image_ids[other_cls_name]
        overlap = set(image_ids) & set(other_image_ids)
        if len(overlap) > 0.1*args.number_of_images:
            print(cls_name, other_cls_name, len(overlap))

device = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = clip.load("ViT-L/14", device=device)
model.to(device)
model.eval()

global_codebook = []
global_codebook_path = os.path.join(f'codebooks/{dataset_name}/codebank', 'global_codebook_embeddings.pth')


if os.path.exists(global_codebook_path):
    print('Codebook already exists, skip generating codebook embeddings.')
else:
    # compute and save codebook text embeddings
    print('Generate Codebook Embeddings....')
    word_list = np.load(f'codebooks/{dataset_name}/codebank/word_list.npy', allow_pickle=True).tolist()

    with torch.no_grad():
        for word in tqdm(word_list):
            texts = [template.format(word) for template in imagenet_templates]
            text_features = clip.tokenize(texts).to(device)
            text_features = model.encode_text(text_features)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = torch.mean(text_features, dim=0)
            text_features = text_features.unsqueeze(0)
            global_codebook.append(text_features)
    global_codebook = torch.concat(global_codebook, dim=0)
    torch.save(global_codebook, global_codebook_path)


# filter codebooks
class ImageDataset_PerClass(Dataset):
    def __init__(self, data_root, image_size, dataset_name, cls_name, number_of_images=500):
        self.data_root = data_root
        self.image_size = image_size
        self.cls_name = cls_name
        self.number_of_images = number_of_images
        self.image_ids = cls_image_ids[cls_name]
        
        _, self.preprocess = clip.load('ViT-L/14', device=device)
        
        cls_name_replaced = cls_name.replace('/', '_')
        imagenet_split_path = f'codebooks/{dataset_name}/imagenet_split/{cls_name_replaced}.txt'
        if args.shots > 0:
            imagenet_split_path = f'codebooks/{dataset_name}/imagenet_{args.shots}shot_split/{cls_name_replaced}.txt'
        with open(imagenet_split_path, 'r') as f:
            self.image_ids = f.readlines()
            self.image_ids = [x.split()[0] for x in self.image_ids][:number_of_images]
            
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_ids = self.image_ids[idx]
        image = Image.open(os.path.join(self.data_root, image_ids))
        image = self.preprocess(image)
        
        return image
    
data_root = ''
global_codebook = torch.load(global_codebook_path).to(device)
class2concepts = {}
os.makedirs(f'codebooks/{dataset_name}/codebank', exist_ok=True)

augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    ..., # add more augmentations here
])

for cls_name in cls_names:
    print(f'Processing {cls_name}..., {cls_names.index(cls_name)+1}/{len(cls_names)}')
    dataset = ImageDataset_PerClass(data_root, 256, dataset_name, cls_name, args.number_of_images)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16)
    
    cls_codebook = []
    token_freq = torch.zeros(global_codebook.size(0)).to(device)
    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images
            images = augment_transform(images).to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            d = torch.mm(image_features, global_codebook.T)
            _, topk_labels = torch.topk(d, k=5)
            topk_index_one_hot = torch.nn.functional.one_hot(topk_labels.view(-1), num_classes=global_codebook.size(0))
            token_freq += torch.sum(topk_index_one_hot, dim=0)
    
    token_freq = np.array(token_freq.cpu().data)
    token_text = np.load(f'codebooks/{dataset_name}/codebank/word_list.npy', allow_pickle=True)
    
    # top-50 concepts with highest frequency
    effective_index = np.argsort(token_freq)[::-1][:50].copy()
    token_text_effective = token_text[effective_index]
    
    concepts = token_text_effective.tolist()
    class2concepts[cls_name] = concepts
    
with open(f'codebooks/{dataset_name}/class2concepts_{args.shots}-imagenet{args.number_of_images}.json', 'w') as f:
    json.dump(class2concepts, f, indent=4)