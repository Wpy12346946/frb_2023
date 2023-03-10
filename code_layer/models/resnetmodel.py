from torchvision.models import resnet152
import torch.nn as nn


class resnet20(nn.Module):
    def __init__(self, index, pretrained=True, **kwargs):
        super(resnet20, self).__init__()
        self.resnet152 = resnet152(pretrained=pretrained, **kwargs)
        self.index = index

    def forward(self, x):
        x = self.resnet152(x)
        x = x[:, self.index]
        return x

class resnet30(nn.Module):
    def __init__(self, index, pretrained=True, **kwargs):
        super(resnet30, self).__init__()
        self.resnet152 = resnet152(pretrained=pretrained, **kwargs)
        self.index = index

    def forward(self, x):
        x = self.resnet152(x)
        x = x[:, self.index]
        return x

"""PyTorch implementation of Wide-ResNet taken from https://github.com/xternalz/WideResNet-pytorch"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out)))
        out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self,  num_classes,inchannels, depth=10, widen_factor=1, dropRate=0,pretrained=False,dataset="cifar10"):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(inchannels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        if dataset == 'cifar10':
            self.blocks = nn.Sequential(
                NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate),
                NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate),
                NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
            )
            self.avg_pool_shape=8
        else:
            self.blocks = nn.Sequential(
                NetworkBlock(4, nChannels[0], nChannels[1], block, 2, dropRate),
                NetworkBlock(4, nChannels[1], nChannels[2], block, 2, dropRate),
                NetworkBlock(3, nChannels[2], nChannels[3], block, 2, dropRate),
                NetworkBlock(3, nChannels[3], 128, block, 2, dropRate),
                NetworkBlock(2, 128, 256, block, 2, dropRate),
                NetworkBlock(2, 256, 512, block, 1, dropRate),
            )
            nChannels[3] = 512
            self.avg_pool_shape=7

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(self.avg_pool_shape)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool2d(out)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def intermediate_forward(self,x,layer_index):
        output, out_features = self.layer_wise_deep_mahalanobis(x)
        return out_features[layer_index]

    def layer_wise_deep_mahalanobis(self,x):
        # focus = [3,7,10,14,17,21,24,28]
        x,layer_output,features = self.forward_layer(x)
        # layers = [layer_output[i] for i in focus]
        layers = layer_output
        return x,layers

    def forward_layer(self,x):
        layer_output=[]
        x = self.conv1(x)
        for ind,layer in enumerate(self.blocks):
            x = layer(x)
            layer_output.append(x)
        x = self.relu(self.bn1(x))
        x = self.avg_pool2d(x)
        layer_output.append(x)
        x = x.view(-1, self.nChannels)
        x = self.fc(x)
        return x,layer_output,self.blocks


class WideResNet_small(nn.Module):
    def __init__(self,  num_classes,inchannels, depth=10, widen_factor=1, dropRate=0,pretrained=False,dataset="cifar10"):
        super(WideResNet_small, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(inchannels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        if dataset == 'cifar10':
            self.blocks = nn.Sequential(
                NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate),
                NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate),
                NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
            )
            self.avg_pool_shape=8
        else:
            self.blocks = nn.Sequential(
                NetworkBlock(2, nChannels[0], nChannels[1], block, 2, dropRate),
                NetworkBlock(2, nChannels[1], nChannels[2], block, 2, dropRate),
                NetworkBlock(2, nChannels[2], nChannels[3], block, 2, dropRate),
                NetworkBlock(1, nChannels[3], 128, block, 2, dropRate),
                NetworkBlock(1, 128, 256, block, 2, dropRate),
                NetworkBlock(1, 256, 512, block, 1, dropRate),
            )
            nChannels[3] = 512
            self.avg_pool_shape=7

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(self.avg_pool_shape)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool2d(out)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def wide_resnet_small(**kwargs):
    model = WideResNet_small(**kwargs)
    return model

def wide_resnet(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model