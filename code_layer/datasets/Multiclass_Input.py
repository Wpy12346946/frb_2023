import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
from attack_methods import attackmethods
import re


def build_multiclassinput_loader(opt, interpretmethodlist, batch_size, train=True, classes=None,
                           num_workers=1, showname=False,pin_memory=True,show_cls_y=False):
    dataset = MultiInputsDataset(opt, interpretmethodlist, train=train, classes=classes,  showname=showname,show_cls_y=show_cls_y)

    if showname:
        _, _, _ = dataset[0]
    elif show_cls_y:
        _,_,_=dataset[0]
    else:
        _, _ = dataset[0]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers,pin_memory=pin_memory)
    return loader, dataset.map_channels


class MultiInputsDataset(Dataset):
    def __init__(self, opt, interpretmethodlist, train, classes=None, showname=False,show_cls_y=False):
        if classes is None:
            self.classes = ['PGD-U', 'DDN-U', 'Clean']
        else:
            self.classes = classes

        self.opt = opt
        self.istrain = train
        self.interpretmethodlist = interpretmethodlist
        self.show_cls_y = show_cls_y

        names = {}
        if opt.attack_box == 'black':
            classifier_net = 'cwnet'
        else:
            classifier_net = opt.classifier_net
        self.classifier_net = classifier_net
        for c in self.classes:
            root = opt.adversarial_root + '{}_{}_{}/{}'.format(opt.dataset, classifier_net, c, 'train' if train else 'test')
            names[c] = [c+n for n in os.listdir(root)]


        self.names = []
        for c in self.classes:
            if c == 'Clean' and len(self.classes) > 3:
                self.names += names[c] * 8

            else:
                self.names += names[c]
            print('{}:{}'.format(c, len(names[c])))

        random.shuffle(self.names)
        self.length = len(self.names)
        self.print_ = False
        self.showname = showname

    def __len__(self):
        return self.length

    def getlabel(self, c):
        if c == 'Clean':
            return 0
        elif c in 'PGD-T'+'PGD-U'+'FGSM-U'+'ADV2-T':
            return 1
        elif c in 'DFool-U'+'CW-U'+'CW-T'+'DDN-U'+'DDN-T':
            return 2
        else:
            return 3

    def __getitem__(self, idx):
        data = [None] * len(self.interpretmethodlist)
        map_channels = [None] * len(self.interpretmethodlist)
        name = self.names[idx]
        cls_y = int(re.findall(r'T(\d+)',name)[0])

        for c in self.classes:
            if c in name:
                label = self.getlabel(c)
                for index, inter in enumerate(self.interpretmethodlist):
                    if inter == 'Data':
                        root = self.opt.adversarial_root + '{}_{}_{}/{}'.format(self.opt.dataset, self.classifier_net, c,
                                                                           'train' if self.istrain else 'test')
                        file_path = os.path.join(root, name.replace(c, ''))
                    else:
                        root = self.opt.saliency_root + '{}_{}_{}/{}/{}'.format(self.opt.dataset,
                                                                                self.classifier_net, c, inter,
                                                                                'train' if self.istrain else 'test')
                        file_path = os.path.join(root, name.replace(c, inter))
                        if 'mnist' in self.opt.dataset and inter=='IG':
                            file_path = os.path.join(root,name.replace(c,'IntGrad'))
                        if 'imagenet' in self.opt.dataset and inter=='IG':
                            file_path = os.path.join(root,name.replace(c,'IntGrad'))

                    data[index] = torch.load(file_path, map_location='cpu')
                    map_channels[index] = data[index].size(0)

        if not self.print_:
            print('Using {} types of Maps ({})'.format(len(self.interpretmethodlist), self.interpretmethodlist))
            print('Map channels are {}; data size is ({},{}); the size of {} is {}.'. \
                  format(map_channels, data[0].size(1), data[0].size(2),
                         'training set' if self.istrain else 'testing set', self.length))
            self.print_ = True

        self.map_channels = map_channels
        if len(self.interpretmethodlist) == 1:
            data = data[0]
            self.map_channels = self.map_channels[0]

        if self.showname:
            return (data, label, self.names[idx])
        elif self.show_cls_y:
            return (data,label,cls_y)
        else:
            return (data, label)