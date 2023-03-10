import torch
from torch.utils.data import Dataset, DataLoader
import random
import os


def build_adversarial_loader(savedpath, batch_size, train=True, num_workers=1, showname=False):
    dataset = AdversarialDataset(savedpath, train=train, transform=None, showname=showname)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    return loader


class AdversarialDataset(Dataset):
    def __init__(self, root_path: str, train: bool, transform=None, showname=False):
        self.data_root = os.path.join(root_path, 'train' if train else 'test')
        self.istrain = train
        self.transform = transform
        self.imagenames = os.listdir(self.data_root)
        self.length = len(self.imagenames)
        self.print_ = False
        self.showname = showname

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_root, self.imagenames[idx])
        data = torch.load(file_path)

        if not self.print_:
            print('data size is {}; {} length is {}.'. \
                  format(data.size(), 'training set' if self.istrain else 'testing set', self.length))
            self.print_ = True

        if self.showname:
            return (data,  self.imagenames[idx])
        else:
            return data
