from torchvision.models import inception_v3
import torch.nn as nn


class incep20(nn.Module):
    def __init__(self, index, pretrained=True, **kwargs):
        super(incep20, self).__init__()
        self.inception_v3 = inception_v3(pretrained=pretrained, **kwargs)
        self.index = index

    def forward(self, x):
        x = self.inception_v3(x)
        x = x[:, self.index]
        return x
