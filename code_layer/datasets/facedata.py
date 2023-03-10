import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random


def build_face_dataloader(datasavedpath, batch_size, shuffle=None, workers=1):
    dataset = datasets.ImageFolder(datasavedpath,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ]))

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
