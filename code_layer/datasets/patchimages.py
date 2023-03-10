import torch
import random
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
random.seed(1000)

def build_patch_loader(datasavedpath, batch_size, train=True, num_workers=1, advonly=False, showpath=False, patchintrain=False):
    transform = {'train': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
                 'test': transforms.Compose([transforms.ToTensor()])}

    #when patching don't use flip
    dataset = SubPatchDataset(datasavedpath, train=train, transform=transform['train' if train else 'test'], advonly=advonly, needpath=showpath)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    return loader

class SubPatchDataset(Dataset):

    def __init__(self, root_path: str, train: bool, transform = None, advonly=False, needpath=False):

        self.path = os.path.join(root_path, 'train' if train else 'test')
        self.istrain = train
        self.transform = transform
        self.needpath = needpath
        self.grayscale = ('mnist' in root_path)

        self.cleannames = []
        self.advernames = []
        self.names = os.listdir(self.path)
        for name in self.names:
            sp = os.path.splitext(name)[0].split('_')
            if len(sp) == 4:
                #raw img
                self.cleannames.append(name)
            else:
                #adv img
                self.advernames.append(name)
        if advonly:
            self.names = self.advernames
        random.shuffle(self.names)
        self.length = len(self.names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sp = os.path.splitext(self.names[idx])[0].split('_')

        target = int(sp[3][1:])
        file_path = os.path.join(self.path, self.names[idx])

        data = Image.open(file_path)
        if self.grayscale:
            data = data.convert('L')
        else:
            data = data.convert('RGB')
        if self.transform is not None:
            data = self.transform(data)
        if self.needpath:
            return (data, target, file_path)
        else:
            return (data, target)

# def build_randsub_loader(datasavedpath, batch_size, train=True, num_workers=1, advonly=False, needpath=False, ispatch=False):
#     transform = {'train': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
#                  'test': transforms.Compose([transforms.ToTensor()])}
#     #when patching don't use flip
#     dataset = RandSubPatchDataset(datasavedpath, train=train, transform=transform['train' if (train and not ispatch) else 'test'], advonly=advonly, needpath=needpath)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
#     return loader
#
# class RandSubPatchDataset(Dataset):
#     """Sub Patch Img Dataset"""
#     def __init__(self, root_path: str, train: bool, transform = None, advonly=False, needpath=False):
#         """
#         Args:
#             root_path(string): path containing all the sub patch image
#             train(bool): whether to load the train dataset or test dataset
#             transform: transform the raw tensors
#         """
#         self.path = os.path.join(root_path, 'train' if train else 'test')
#         self.istrain = train
#         self.transform = transform
#         self.needpath = needpath
#
#         self.cleannames = []
#         self.advernames = []
#         self.names = os.listdir(self.path)
#         for name in self.names:
#             sp = os.path.splitext(name)[0].split('_')
#             if len(sp) == 5:
#                 #raw img
#                 self.cleannames.append(name)
#             else:
#                 #adv img
#                 self.advernames.append(name)
#         if advonly:
#             self.names = self.advernames
#         random.shuffle(self.names)
#         self.length = len(self.names)
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, idx):
#         sp = os.path.splitext(self.names[idx])[0].split('_')
#         # if len(sp) == 5:
#         #     target = 0
#         # else:
#         #     target = 1
#         target = int(sp[3][1:])
#         file_path = os.path.join(self.path, self.names[idx])
#         # data = mpimg.imread(file_path)
#         data = Image.open(file_path)
#         data = data.convert('RGB')
#         if self.transform is not None:
#             data = self.transform(data)
#         if self.needpath:
#             return (data, target, file_path)
#         else:
#             return (data, target)