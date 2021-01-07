import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

def read_off(filename):
    xyz = []
    with open(filename, 'r') as f:
        header = f.readline().split()
        if 'OFF' != header[0]:
            n_vertices = int(header[0][3:])
        else:
            n_vertices, n_faces, n_edges = list(map(int, f.readline().split()))

        for _ in range(n_vertices):
            xyz.append(list(map(float, f.readline().split())))

    xyz = np.array(xyz, dtype='f4')

    assert xyz.shape[1] == 3

    return xyz


# class ModelNet40Dataset(Dataset):
#
#     def __init__(self, root='./dataset/ModelNet40', npoints=1024, split='train'):
#         super(ModelNet40Dataset, self).__init__()
#         self.npoints = npoints
#         self.split = split
#         self.df = pd.read_csv(os.path.join(root, f'{split}_list.csv'))
#
#     def __getitem__(self, idx):
#         filename, y = self.df.loc[idx][1:]
#         y = np.array(y)
#         x = read_off(filename)
#         indices = np.random.choice(np.arange(x.shape[0]), self.npoints, replace=True)
#         x = x[indices]
#         return x, y
#
#     def __len__(self):
#         return len(self.df)

class ModelNet40Dataset(Dataset):

    def __init__(self, root='./dataset/ModelNet40/Tensor', split='train'):
        self.root = root
        self.split = split
        self.file_list = [file for file in os.listdir(os.path.join(root, split)) if file[0] != '.']

    def __getitem__(self, idx):
        x, y = torch.load(os.path.join(self.root, self.split, self.file_list[idx]))
        return x, y

    def __len__(self):
        return len(self.file_list)

class DatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_ratio):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_ratio = valid_ratio

    def get_dataloader(self, split='train'):
        if split=='train':
            return self.get_train_valid_dataloaders()
        else:
            return self.get_testloader()

    def get_train_valid_dataloaders(self):
        ds = ModelNet40Dataset(split='train')
        num_data = len(ds)
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        split = int(np.floor(self.valid_ratio * num_data))
        train_indices, valid_indices = indices[split:], indices[:split]
        train_sampler, valid_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(valid_indices)

        train_loader = DataLoader(ds, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        valid_loader = DataLoader(ds, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader

    def get_test_dataloader(self):
        ds = ModelNet40Dataset(split='test')
        indices = np.arange(len(ds))
        np.random.shuffle(indices)

        test_sampler = SubsetRandomSampler(indices)
        test_loader = DataLoader(ds, batch_size=self.batch_size, sampler=test_sampler,
                                 num_workers=self.num_workers)
        return test_loader

if __name__ == '__main__':
    ds = ModelNet40Dataset()
    x, y = ds[0]
    x.astype('f4')
    print(x.dtype)