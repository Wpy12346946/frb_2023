import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import re


def adver_train_loader(cleanpath,advpath, batch_size, train=True, workers=1, showname=False):
    dataset = AdverTrainDataset(cleanpath,advpath, train=train, transform=None, showname=showname)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=workers)
    return loader


class AdverTrainDataset(Dataset):
    def __init__(self, clean_path: str,adv_path:str, train: bool, transform=None, showname=False):
        self.clean_root = os.path.join(clean_path, 'train' if train else 'test')
        self.adv_root=os.path.join(adv_path,'train'if train else 'test')
        self.istrain = train
        self.transform = transform
        self.imagenames_clean = os.listdir(self.clean_root)
        # self.imagenames_adv = os.listdir(self.adv_root)
        self.length = len(self.imagenames_clean)
        print(len(self.imagenames_clean))
        # print(len(self.imagenames_adv))
        self.print_ = False
        self.showname = showname

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_root, self.imagenames_clean[idx])
        clean_data = torch.load(clean_path)
        # adv_path = os.path.join(self.adv_root,self.imagenames_adv[idx])
        # adv_data = torch.load(adv_path)
        r=re.match(r'.*T(\d+).*',self.imagenames_clean[idx])
        label=int(r.group(1))

        if not self.print_:
            print('data size is {}; {} length is {}.'. \
                  format(clean_data.size(), 'training set' if self.istrain else 'testing set', self.length))
            self.print_ = True

        if self.showname:
            # return (clean_data, adv_data, label, self.imagenames[idx])
            return (clean_data,label,self.imagenames_clean[idx])
        else:
            # return (clean_data, adv_data, label)
            return(clean_data,label)

if __name__=='__main__':
    s='image_E0_N0_T17.pth'
    r=re.match(r'.*T(\d+).*',s)
    label=int(r.group(1))
    print(label)

    path='../../adversarial_data/mnist_simple_net_PGD-U/test'
    listdir=os.listdir(path)
    print(listdir)