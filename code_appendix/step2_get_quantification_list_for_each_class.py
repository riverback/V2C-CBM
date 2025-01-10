import clip
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CUB')
parser.add_argument('--shots', type=int, default=0, help='0 for no shots')
args = parser.parse_args()
print(args.__dict__)

dataset_name = args.dataset
cls_names = np.load(f'codebooks/{dataset_name}/cls_names.npy', allow_pickle=True)
cls_names = cls_names.tolist()
if args.shots > 0:
    cls_name_embeddings = np.load(f'codebooks/{dataset_name}/cls_{args.shots}shot_images_embedding.npy', allow_pickle=True).item()
else:
    cls_name_embeddings = np.load(f'codebooks/{dataset_name}/cls_name_embedding.npy', allow_pickle=True).item()


imagenet_image_root = ''
with open('imagenet_split/train/class_labels.txt', 'r') as f:
    imagenet_train_image_label_ids = f.readlines()
    imagenet_train_image_label_ids = [x.strip() for x in imagenet_train_image_label_ids]
    
N = 1000 # unlabeled imageset size
imagenet_train_image_ids = []
imagenet_train_image_ids = random.sample(imagenet_train_image_label_ids, N)
with open(f'codebooks/{dataset_name}/imagenet_train_image_ids.txt', 'w') as f:
    for image_id in imagenet_train_image_ids:
        f.write(f'{image_id}\n')

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
model.to(device)
model.eval()

class ImageNetDataset(Dataset):
    def __init__(self, image_root, image_ids, preprocess):
        self.image_root = image_root
        self.image_ids = image_ids
        self.preprocess = preprocess

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_root, image_id)        
        image = self.preprocess(Image.open(image_path))
        return image, idx

    def __len__(self):
        return len(self.image_ids)

imagenet_dataset = ImageNetDataset(imagenet_image_root, imagenet_train_image_ids, preprocess)
imagenet_dataloader = DataLoader(imagenet_dataset, batch_size=128, shuffle=False, num_workers=16)

cls_image_ids = {}
if args.shots > 0:
    os.makedirs(f'codebooks/{dataset_name}/imagenet_{args.shots}shot_split', exist_ok=True)
else:
    os.makedirs(f'codebooks/{dataset_name}/imagenet_split', exist_ok=True)
print('number of classes:', len(cls_names))
for cls_name in cls_names:
    flag = True if '/' in cls_name else False
    save_cls_name = cls_name.replace('/', '_') if flag else cls_name
    if args.shots > 0:
        if os.path.exists(f'codebooks/{dataset_name}/imagenet_{args.shots}shot_split/{save_cls_name}.txt'):
            print(f'{cls_name} already exists')
            continue
    else:
        if os.path.exists(f'codebooks/{dataset_name}/imagenet_split/{save_cls_name}.txt'):
            print(f'{cls_name} already exists')
            continue
    cls_image_ids[cls_name] = []
    
    cls_name_embedding = cls_name_embeddings[cls_name]
    cls_name_embedding = torch.from_numpy(cls_name_embedding).to(device)
    with torch.no_grad():
        for images, idxs in tqdm(imagenet_dataloader):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ cls_name_embedding.T
            similarities = torch.mean(similarities, dim=-1)                
            for i, idx in enumerate(idxs):
                cls_image_ids[cls_name].append((imagenet_train_image_ids[idx], similarities[i].item()))
    cls_image_ids[cls_name] = sorted(cls_image_ids[cls_name], key=lambda x: x[1], reverse=True)
    
    if args.shots > 0:
        with open(f'codebooks/{dataset_name}/imagenet_{args.shots}shot_split/{save_cls_name}.txt', 'w') as f:
            for image_id, similarity in cls_image_ids[cls_name]:
                f.write(f'{image_id} {similarity}\n')
    else:
        with open(f'codebooks/{dataset_name}/imagenet_split/{save_cls_name}.txt', 'w') as f:
            for image_id, similarity in cls_image_ids[cls_name]:
                f.write(f'{image_id} {similarity}\n')
    print(f'{cls_name} done, {cls_names.index(cls_name)+1}/{len(cls_names)}')