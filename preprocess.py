import os
import pandas as pd
import torch
import numpy as np
import pickle
from dataloader import read_off
from tqdm import tqdm

def make_dict(root='./dataset/ModelNet40'):
    cls_dict = {}
    idx = 0
    dirs = os.listdir(root)

    for dir in dirs:
        if dir[0] == '.':
            continue
        cls_dict[dir] = idx
        idx += 1

    with open(os.path.join(root, 'cls_dict.pkl'), 'wb') as f:
        pickle.dump(cls_dict, f)

def make_csv(root='./dataset/ModelNet40', split='train'):
    df = pd.DataFrame(columns=['path', 'label'])
    dirs = os.listdir(root)

    with open(os.path.join(root, 'cls_dict.pkl'), 'rb') as f:
        cls_dict = pickle.load(f)

    for dir in dirs:
        if dir[0] == '.' or os.path.isdir(os.path.join(root, dir)) == False:
            continue

        path = os.path.join(root, dir)
        for filename in os.listdir(os.path.join(path, split)):
            assert filename[-3:] == 'off'
            df = df.append({'path': os.path.join(path, split, filename),
                            'label': cls_dict[dir]}, ignore_index=True)

    df.to_csv(f'./dataset/{split}_list.csv')

def make_tensor_file(root='./dataset/ModelNet40', npoints=1024):
    train_df = pd.read_csv(os.path.join(root, 'train_list.csv'))
    test_df = pd.read_csv(os.path.join(root, 'test_list.csv'))

    ################ Train Dataset ################
    pbar = tqdm(range(len(train_df)))
    for i in pbar:
        filename, y = train_df.loc[i][1:]
        y = np.array(y)
        x = read_off(filename)
        indices = np.random.choice(np.arange(x.shape[0]), npoints, replace=True)
        x = x[indices]
        sample = (x, y)
        filename = os.path.split(filename)[1][:-4]
        torch.save(sample, f'./dataset/ModelNet40/Tensor/train/{filename}_{npoints}')
    ################################################

    ################## Test Dataset ################
    pbar = tqdm(range(len(test_df)))
    for i in pbar:
        filename, y = test_df.loc[i][1:]
        y = np.array(y)
        x = read_off(filename)
        indices = np.random.choice(np.arange(x.shape[0]), npoints, replace=True)
        x = x[indices]
        sample = (x, y)
        filename = os.path.split(filename)[1][:-4]
        torch.save(sample, f'./dataset/ModelNet40/Tensor/test/{filename}_{npoints}')
    ################################################

def make_tensor_list():
    train_df = pd.DataFrame(columns=['filename'])
    test_df = pd.DataFrame(columns=['filename'])



if __name__ == '__main__':

    if os.path.isfile('./dataset/ModelNet40/cls_dict.pkl') == False:
        make_dict()
    if os.path.isfile('./dataset/ModelNet40/train_list.csv') == False:
        make_csv(split='train')

    if os.path.isfile('./dataset/ModelNet40/test_list.csv') == False:
        make_csv(split='test')

    make_tensor_file(npoints=1024)