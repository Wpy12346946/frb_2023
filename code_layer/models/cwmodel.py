import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class CWNet(nn.Module):
    def __init__(self, features, num_classes=10, input_channels=3, init_weights=True,**kwargs):
        super(CWNet, self).__init__()
        self.features = features

        if input_channels==3:
            size =8
            inc = 128
            fc = 256
        else:
            size = 7
            inc = 64
            fc = 200

        self.classifier = nn.Sequential(
            nn.Linear(inc * size * size, fc),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fc, fc),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fc, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        b, c, w, h = x.size()
        x = x.view(b, -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, input_channels=3):
    layers = []
    in_channels = input_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def cwnet(pretrained=False, inchannels=3,dataset='cifar10', **kwargs):
    layers_mn = [32, 32, 'M', 64, 64, 'M']
    layers_cf = [64, 64, 'M', 128, 128, 'M']

    if pretrained:
        kwargs['init_weights'] = False
    if inchannels == 3:
        model = CWNet(make_layers(layers_cf, batch_norm=True, input_channels=inchannels), input_channels=inchannels, **kwargs)
    else:
        model = CWNet(make_layers(layers_mn, batch_norm=True, input_channels=inchannels), input_channels=inchannels, **kwargs)
    return model
