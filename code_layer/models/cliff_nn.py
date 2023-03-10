import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class MyCliff(nn.Module):
    def __init__(self, k=1):
        super(MyCliff, self).__init__()
        self.k = k
        # self.act = nn.ReLU(False)
        # self.act = nn.Sigmoid()

    def forward(self, input):
        return torch.clamp(self.k*input,min=0,max=1)
        # return torch.clamp(self.k*input, min=-0.5,max=0.5) + 0.5
        # return self.act(input)

class CliffNN(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(CliffNN, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            # nn.ReLU(False),
            MyCliff(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            # nn.ReLU(False),
            MyCliff(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        # x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_k(self,k):
        for m in self.modules():
            if isinstance(m,MyCliff):
                m.k = k

    def step_k(self):
        for m in self.modules():
            if isinstance(m,MyCliff):
                m.k = m.k+1
    
    def forward_layer(self,x):
        layer_output=[]
        layer_names = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer,MyCliff):
                layer_output.append(x)
                layer_names.append(layer)
        # x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x,layer_output,layer_names

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
                # layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                layers += [conv2d,nn.BatchNorm2d(v),MyCliff()]
            else:
                # layers += [conv2d, nn.ReLU(inplace=True)]
                layers += [conv2d, MyCliff()]
            in_channels = v
    return nn.ModuleList(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'A1': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def clf11bn(pretrained=False, inchannels=3,dataset='cifar10', **kwargs):
    if 'mnist' in dataset:
        phase='A1'
    else:
        phase='A'
    model = CliffNN(make_layers(cfg[phase], batch_norm=True, input_channels=inchannels), **kwargs)
    return model
