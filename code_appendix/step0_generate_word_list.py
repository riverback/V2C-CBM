import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10')
args = parser.parse_args()

dataset_name = args.dataset
save_path = f'codebooks/{dataset_name}/codebank'

# the code list comes from https://norvig.com/ngrams/count_1w.txt

with open(f'codebooks/{dataset_name}/codebank/adj_list.txt', 'r') as f:
    adj_list = f.readlines()
    adj_list = [x.strip() for x in adj_list]
    adj_list = list(set(adj_list))
with open(f'codebooks/{dataset_name}/codebank/noun_list.txt', 'r') as f:
    noun_list = f.readlines()
    noun_list = [x.strip() for x in noun_list]
    noun_list = list(set(noun_list))
    
relationships = ['has', 'part of', 'at location', 'made of']

texts = []

# atomic
texts.extend(adj_list)
texts.extend(noun_list)

# bigram
for adj in adj_list:
    for noun in noun_list:
        texts.append(f'{adj} {noun}')

# trigram
for rel in relationships:
    for adj in adj_list:
        for noun in noun_list:
            texts.append(f'{rel} {adj} {noun}')

texts = list(set(texts))

print('Number of texts:', len(texts))

with open(os.path.join(save_path, 'word_list.txt'), 'w') as f:
    for text in texts:
        f.write(f'{text}\n')
np.save(os.path.join(save_path, 'word_list.npy'), texts)