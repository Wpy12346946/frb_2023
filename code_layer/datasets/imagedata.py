import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

from code_layer.face.utils.dataloader import Arcface

# 30 objects selected from imagenet
IMAGEINFO = {'n01514668': 7, 'n01641577': 30, 'n01667778': 36, 'n01737021': 58, 'n01774384': 75,
             'n01871265': 101, 'n02088238': 161, 'n02123159': 282, 'n02666196': 398, 'n02879718': 456,
             'n03026506': 496, 'n03095699': 510, 'n03291819': 549, 'n03642806': 620, 'n03876231': 696,
             'n03977966': 734, 'n04039381': 752, 'n04592741': 908, 'n07753592': 954, 'n09472597': 980,
             'n04111531': 766,
             'n03721384': 642,
             'n04204238': 790,
             'n03887697': 700,
             'n03131574': 520,
             'n02676566': 402,
             'n01443537': 1,
             'n04467665': 867,
             'n04069434': 759,
             'n04356056': 837,
             }

IMAGEFILE = list(IMAGEINFO.keys())
IMAGEFILE.sort()
IMAGEINDEX = list(IMAGEINFO.values())
IMAGEINDEX.sort()

l = {}

datasetnames = ['mnist', 'cifar10', 'imagenet', 'fmnist']


def build_loader(datasavedpath, datasetname, batch_size, train=True, shuffle=None, workers=1):
    # implemented by Pytorch
    cifar10 = datasets.CIFAR10
    mnist = datasets.MNIST
    fmnist = datasets.FashionMNIST
    # implemented by ourselves
    imagenet = ImageNetData
    arcface = Arcface

    transforms_ = {'cifar10': {'train': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
                               'test': transforms.Compose([transforms.ToTensor()])},

                   'mnist': {'train': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
                             'test': transforms.Compose([transforms.ToTensor()])},

                   'fmnist': {'train': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
                              'test': transforms.Compose([transforms.ToTensor()])},

                   'imagenet': {'train': transforms.Compose(
                       [transforms.RandomHorizontalFlip(), transforms.Resize(256, Image.BICUBIC),
                        transforms.CenterCrop(224), transforms.ToTensor()]),
                       'test': transforms.Compose([transforms.Resize(256, Image.BICUBIC),
                                                   transforms.CenterCrop(224), transforms.ToTensor()])},
                   'arcface': {
                       'train': transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.Resize(112, Image.BICUBIC),
                           transforms.CenterCrop(112), transforms.ToTensor()]),

                       'test': transforms.Compose([
                           transforms.Resize(112, Image.BICUBIC),
                           transforms.CenterCrop(112), transforms.ToTensor()])}
                   }

    # Using dataset of  'datasetname'
    dataset = eval('arcface')(os.path.join(datasavedpath, datasetname), train=train, download=False,
                                transform=transforms_['arcface']['train' if train else 'test'])

    data, _ = dataset[0]
    print('The size of dataset({}) is {}; the length of {} is {}.\n'. \
          format(datasetname, data.size(), 'training set' if train else 'testset', dataset.__len__()))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    return loader


class ImageNetData(Dataset):
    def __init__(self, root_path: str, train, download=None, transform=None):
        self.root = root_path
        self.istrain = train
        self.transform = transform
        self.imageinfor = {}

        index = 0
        print('using imagenet:{}'.format(IMAGEFILE))
        for file in IMAGEFILE:
            imagepath = os.path.join(self.root, file, 'train' if train else 'test')
            imagenames = os.listdir(imagepath)
            for name in imagenames:
                path = os.path.join(imagepath, name)
                self.imageinfor[path] = index
            index += 1

        self.all_imagepath = list(self.imageinfor.keys())
        random.shuffle(self.all_imagepath)

    def __len__(self):
        return len(self.all_imagepath)

    def __getitem__(self, item):
        path = self.all_imagepath[item]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(path)
            l[path] = e
            raise e
        label = self.imageinfor[path]

        if self.transform is not None:
            image = self.transform(image)

        return (image, label)


# def get_original_image(imagetensor):
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     mean = torch.tensor(mean).unsqueeze_(-1).unsqueeze_(-1)
#     std = torch.tensor(std).unsqueeze_(-1).unsqueeze_(-1)
#     return imagetensor * std + mean

if __name__ == '__main__':
    import warnings
    from tqdm import tqdm

    warnings.filterwarnings("error", category=UserWarning)
    root_path = '../../data/imagenet'
    imgnet_train = ImageNetData(root_path, True)
    for i in tqdm(range(len(imgnet_train))):
        try:
            _, _ = imgnet_train.__getitem__(i)
        except Exception as e:
            pass
    with open('out.txt', 'w') as f:
        print(l, file=f)
