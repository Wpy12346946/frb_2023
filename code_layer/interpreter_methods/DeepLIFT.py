import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import DeepLift

def attribute_image_features(algorithm, input, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=labels[ind],
                                              **kwargs
                                             )
    
    return tensor_attributions

class DL():
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.model_output = None
        self.model_pred = None
        self.model.eval()

        self.dl = DeepLift(model)

    def interpret(self, x):
        x.requires_grad=True
        with torch.enable_grad():
            self.model.zero_grad()
            self.model_output = self.model(x)
            self.model_pred = self.model_output.max(1)[1]
            ret = self.dl.attribute(x, target=self.model_pred, baselines=x * 0.1)
            return ret.detach()

    def release(self):
        '''
        释放hook和内存，每次计算saliency后都要调用release()
        :return:
        '''
        for handle in self.handles:
            handle.remove()
        # self.model.zero_grad()
        for p in self.model.parameters():
            del p.grad
            p.grad=None
        self.handles = []
        self.model_output = None
        self.model_pred = None
