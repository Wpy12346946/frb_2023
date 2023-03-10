import torch
import torch.nn as nn
import torch.nn.functional as F

class global_tree_differ(nn.Module):
    def __init__(self, model_net, input_channel, output_classes):
        super(global_tree_differ, self).__init__()

    def forward(self, x):
        return self.simclr_model(x)
