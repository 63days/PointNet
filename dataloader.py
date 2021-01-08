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

class ModelNetDataset(Dataset):

    def __init__(self, root, cls_dict, npoints=1024, split='train'):
        self.root = root
        self.cls_dict = cls_dict
        self.npoints = npoints
        self.split = split
        self.filelist = np.genfromtxt(os.path.join(root, f'modelnet40_{split}.txt'), dtype=np.str)

    def __getitem__(self, idx):
        filename = self.filelist[idx]
        dirname = filename[:-5]

        pc = np.genfromtxt(os.path.join(self.root, dirname, filename, '.txt'), delimiter=',')[:, :3]
        indices = np.random.choice(pc.shape[0], self.npoints, replace=True)
        pc = pc[indices]
        pc = self._normalize(pc)

        label = self.cls_dict[dirname]

        return pc, label

    def __len__(self):
        return len(self.filelist)

    def _normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc -= centroid
        dist = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc /= dist
        return pc




class DatasetWrapper(object):
    def __init__(self, cls_dict, batch_size, num_workers, valid_ratio):
        self.cls_dict = cls_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_ratio = valid_ratio

    def get_dataloader(self, split='train'):
        if split=='train':
            return self.get_train_valid_dataloaders()
        else:
            return self.get_testloader()

    def get_train_valid_dataloaders(self):
        ds = ModelNetDataset(cls_dict=self.cls_dict, split='train')
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
        ds = ModelNetDataset(cls_dict=self.cls_dict, split='test')
        indices = np.arange(len(ds))
        np.random.shuffle(indices)

        test_sampler = SubsetRandomSampler(indices)
        test_loader = DataLoader(ds, batch_size=self.batch_size, sampler=test_sampler,
                                 num_workers=self.num_workers)
        return test_loader

if __name__ == '__main__':
    ds = ModelNetDataset()
    x, y = ds[0]
    x.astype('f4')
    print(x.dtype)