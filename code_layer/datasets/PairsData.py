import torch
from torch.utils.data import Dataset, DataLoader
import random
import os


def pairs_loader(savedpath, batch_size, workers=0,shuffle=True):
    dataset = PairsDataset(savedpath, transform=None)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    return loader


class PairsDataset(Dataset):
    def __init__(self, root_path: str, transform=None):
        self.file_path = root_path
        self.transform = transform
        self.dataset=torch.load(self.file_path)
        self.length = len(self.dataset)
        self.print_ = False

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.dataset[idx]

        if not self.print_:
            print('data size is {};length is {}.'. \
                  format(data['Org'].size(), self.length))
            self.print_ = True

        return data['Org'],data['Org_label'],data['Pair'],data['Pair_label']
