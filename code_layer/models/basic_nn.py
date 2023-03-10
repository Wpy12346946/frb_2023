import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class BasicNN(nn.Module):
    def __init__(self, input_size, num_classes, init_weights=True,fc=256):
        super(BasicNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, fc),
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
