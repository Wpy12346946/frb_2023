import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
from attack_methods import attackmethods


def build_maps_loader(adversarialroot, saliencyroot, interpretmethod, batch_size, train=True, howtousedata=2,
                      num_workers=1, advonly=2, showname=False, samenum=False):
    
    dataset = MapsDataset(adversarialroot, saliencyroot, interpretmethod, train=train, howtousedata=howtousedata,
                          advonly=advonly, showname=showname, samenum=samenum)

    print("How to use data: {}".format(howtousedata))

    if showname:
        map_ , _ , _ = dataset[0]
    else:
        map_, _ = dataset[0]
    map_channels = map_.size(0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    return loader, map_channels


class MapsDataset(Dataset):
    def __init__(self, adversarialroot, saliencyroot, interpretmethod, train,
                 howtousedata, advonly=2, showname=False, samenum=False):

        self.adver_data_root = os.path.join(adversarialroot, 'train' if train else 'test')
        self.adver_saliency_root = os.path.join(saliencyroot, 'train' if train else 'test')
        for attack in attackmethods:
            if attack in adversarialroot:
                self.clean_data_root = self.adver_data_root.replace(attack, 'Clean')
                self.clean_saliency_root = self.adver_saliency_root.replace(attack, 'Clean')

        self.istrain = train
        self.howtousedata = howtousedata
        self.interpretmethod = interpretmethod
        if interpretmethod=='IG':
            self.interpretmethod='IntGrad'

        if advonly == 0:
            cleannames = os.listdir(self.clean_data_root)
            self.datanames = cleannames
        elif advonly == 1:
            advernames = os.listdir(self.adver_data_root)
            self.datanames = advernames
        else:
            cleannames = os.listdir(self.clean_data_root)
            advernames = os.listdir(self.adver_data_root)
            cleannum = len(cleannames)
            advernum = len(advernames)
            if samenum:
                num = min(cleannum, advernum)
                self.datanames = cleannames[:num] + advernames[:num]
            else:
                self.datanames = cleannames+ advernames

        random.shuffle(self.datanames)
        self.length = len(self.datanames)
        self.print_ = False
        self.showname = showname

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sp = self.datanames[idx].split('_')
        if len(sp) == 5:
            data_path = os.path.join(self.adver_data_root, self.datanames[idx])
            saliency_path = os.path.join(self.adver_saliency_root, self.interpretmethod+self.datanames[idx])
            label = 0
        else:
            data_path = os.path.join(self.clean_data_root, self.datanames[idx])
            saliency_path = os.path.join(self.clean_saliency_root, self.interpretmethod+self.datanames[idx])
            label = 1

        if self.howtousedata == 0:
            map = torch.load(data_path)
        elif self.howtousedata == 1:
            map = torch.load(saliency_path)
        else:
            data = torch.load(data_path)
            saliency = torch.load(saliency_path)
            map = torch.cat([data, saliency], 0)

        if not self.print_:
            print('Map size is {}; {} length is {}.'. \
                  format(map.size(), 'training set' if self.istrain else 'testing set', self.length))
            self.print_ = True

        if self.showname:
            return (map, label, self.datanames[idx])
        else:
            return (map, label)


def build_multimaps_loader(opt, interpretmethodlist, batch_size, train=True,
                      num_workers=1, advonly=2, showname=False):
    dataset = MultiMapsDataset(opt, interpretmethodlist, train=train, advonly=advonly, showname=showname)

    if showname:
        _, _, _ = dataset[0]
    else:
        _, _ = dataset[0]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    return loader, dataset.map_channels


class MultiMapsDataset(Dataset):
    def __init__(self, opt, interpretmethodlist, train, advonly=2, showname=False):
        self.adversarialroots = []
        self.cleanroots = []
        self.index = None
        for i, inter in enumerate(interpretmethodlist):
            if inter == 'Data':
                self.index = i
                root=opt.adversarial_root + '{}_{}_{}/'.format(opt.dataset, opt.classifier_net, opt.attack)
            else:
                root=opt.saliency_root + '{}_{}_{}/{}/'.format(opt.dataset, opt.classifier_net, opt.attack, inter)
            root = os.path.join(root, 'train' if train else 'test')
            self.adversarialroots.append(root)

            for attack in attackmethods:
                if attack in root:
                    self.cleanroots.append(root.replace(attack, 'Clean'))

        self.istrain = train
        self.interpretmethodlist = interpretmethodlist

        if advonly == 0:
            cleannames = os.listdir(self.cleanroots[self.index])
            self.datanames = cleannames
        elif advonly == 1:
            advernames = os.listdir(self.adversarialroots[self.index])
            self.datanames = advernames
        else:
            cleannames = os.listdir(self.cleanroots[self.index])
            advernames = os.listdir(self.adversarialroots[self.index])
            self.datanames = cleannames + advernames

        random.shuffle(self.datanames)
        self.length = len(self.datanames)
        self.print_ = False
        self.showname = showname

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sp = self.datanames[idx].split('_')
        data = [None] * len(self.interpretmethodlist)
        map_channels = [None] * len(self.interpretmethodlist)
        if len(sp) == 5:
            for index,inter in enumerate(self.interpretmethodlist):
                if index==self.index:
                    file_path = os.path.join(self.adversarialroots[index], self.datanames[idx])
                else:
                    file_path = os.path.join(self.adversarialroots[index], inter+self.datanames[idx])
                data[index] = torch.load(file_path)
                map_channels[index] = data[index].size(0)
            label = 0
        else:
            for index, inter in enumerate(self.interpretmethodlist):
                if index == self.index:
                    file_path = os.path.join(self.cleanroots[index], self.datanames[idx])
                else:
                    file_path = os.path.join(self.cleanroots[index], inter + self.datanames[idx])
                data[index] = torch.load(file_path)
                map_channels[index] = data[index].size(0)
            label = 1

        if not self.print_:
            print('Using {} types of Maps ({})'.format(len(self.interpretmethodlist), self.interpretmethodlist))
            print('Map channels are {}; data size is ({},{}); the size of {} is {}.'. \
                  format(map_channels, data[0].size(1), data[0].size(2),'training set' if self.istrain else 'testing set', self.length))
            self.print_ = True

        self.map_channels = map_channels

        if self.showname:
            return (data, label, self.datanames[idx])
        else:
            return (data, label)