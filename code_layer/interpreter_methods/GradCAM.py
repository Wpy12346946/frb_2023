import torch
import torch.nn as nn
import torch.nn.functional as F
from .InterpreterBase import Interpreter


class GradCAM(Interpreter):
    def __init__(self, model, specific_layer=None):
        super(GradCAM, self).__init__(model)
        self.last_conv_layer = self.search_convlayers(specific_layer)
        self.last_conv_layer_grad = None
        self.last_conv_layer_feature = None
        self.register_hook()

    def search_convlayers(self, specific_layer):
        if specific_layer is None:
            convlayers = []
            for layer in self.model.named_modules():
                if isinstance(layer[1], nn.Conv2d):
                    convlayers.append(layer)
            return convlayers[-1]

        else:
            for layer in self.model.named_modules():
                if layer[0]==specific_layer:
                    return layer

    def register_hook(self):
        def forward_hook(module, input, output):
            self.last_conv_layer_feature = output

        def backward_hook(module, grad_in, grad_out):
            self.last_conv_layer_grad = grad_out[0]

        for layer in self.model.named_modules():
            if layer[0]==self.last_conv_layer[0]:
                self.handles.append(layer[1].register_forward_hook(forward_hook))
                self.handles.append(layer[1].register_backward_hook(backward_hook))

    def interpret(self, x):
        b, c, w, h = x.size()
        self.model_output = self.model(x)
        self.model.zero_grad()
        self.model_pred = self.model_output.max(1)[1]

        one_hot = torch.zeros_like(self.model_output, dtype=torch.long).scatter_(1, self.model_pred.unsqueeze(1), 1)
        one_hot = torch.sum(one_hot.float() * self.model_output)
        one_hot.backward(retain_graph=True)
        # one_hot.backward()
        # del one_hot

        last_conv_layer_feature = self.last_conv_layer_feature
        last_conv_layer_grad = self.last_conv_layer_grad

        gcam_weights = torch.mean(last_conv_layer_grad, dim=[2,3])

        gcam = gcam_weights.unsqueeze(-1).unsqueeze(-1).expand(last_conv_layer_feature.size()) * last_conv_layer_feature
        gcam = torch.clamp(torch.sum(gcam, dim=1), min=0)
        gcam = gcam.unsqueeze(1)
        # print("cam size", cam.size())

        gcam = F.interpolate(gcam,size=[w,h],mode='bilinear', align_corners=False)
        # print("cam size", cam.size())
        return gcam

    def release(self):
        super(GradCAM, self).release()
        self.last_conv_layer = None
        self.last_conv_layer_grad = None
        self.last_conv_layer_feature = None