import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = [
    'VGG', 'vgg11', 'vgg11bn', 'vgg13', 'vgg13bn', 'vgg16', 'vgg16bn',
    'vgg19bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            # nn.ReLU(False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # nn.ReLU(False),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        for ind,layer in enumerate(self.features):
            # print(x.shape,layer)
            x = layer(x)
        # x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def forward_layer(self,x):
        '''
        ModuleList(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (6): ReLU(inplace=True)
            (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (9): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (10): ReLU(inplace=True)
            (11): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (13): ReLU(inplace=True)
            (14): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (15): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (16): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (17): ReLU(inplace=True)
            (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (19): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (20): ReLU(inplace=True)
            (21): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (22): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (23): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (24): ReLU(inplace=True)
            (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (26): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (27): ReLU(inplace=True)
            (28): AvgPool2d(kernel_size=2, stride=2, padding=0)
            )
        '''
        layer_output=[]
        for ind,layer in enumerate(self.features):
            x = layer(x)
            layer_output.append(x)
        # x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x,layer_output,self.features

    def intermediate_forward(self,x,layer_index):
        output, out_features = self.layer_wise_deep_mahalanobis(x)
        return out_features[layer_index]

    def layer_wise_deep_mahalanobis(self,x):
        focus = [3,7,10,14,17,21,24,28]
        x,layer_output,features = self.forward_layer(x)
        layers = [layer_output[i] for i in focus]
        return x,layers

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
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers += [nn.AvgPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.ModuleList(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'A1': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, inchannels=3,dataset='cifar10', **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    print(dataset)
    if 'mnist' in dataset:
        phase='A1'
    else:
        phase='A'
    model = VGG(make_layers(cfg[phase], batch_norm=False, input_channels=inchannels), **kwargs)
    return model



def vgg11bn(pretrained=False, inchannels=3,dataset='cifar10', **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    print(dataset)
    if 'mnist' in dataset:
        phase='A1'
    else:
        phase='A'
    model = VGG(make_layers(cfg[phase], batch_norm=True, input_channels=inchannels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, inchannels=3, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], input_channels=inchannels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13bn(pretrained=False, inchannels=3, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True, input_channels=inchannels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, inchannels=3, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], input_channels=inchannels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16bn(pretrained=False, inchannels=3, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True, input_channels=inchannels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, inchannels=3, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], input_channels=inchannels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19bn(pretrained=False, inchannels=3, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True, input_channels=inchannels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

if __name__ == '__main__':
    model = vgg11bn()
    print(type(model.features[27]))
    l = [l for ind,(n,l) in enumerate(model.named_modules())]
    for i in l:
        print(i)
    
    inp = torch.randn((10,3,32,32))
    y,layer_output,features = model.forward_layer(inp)
    # print([l.shape for l in layer_output])