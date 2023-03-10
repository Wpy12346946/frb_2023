import torch
import torch.nn as nn
from .InterpreterBase import Interpreter

class LayerWise_VG(Interpreter):
    def __init__(self, model,interesting = [2,6,10,13,17,20,24,27]):
        super(LayerWise_VG, self).__init__(model)
        self.grads = []
        self.handles = []
        self.interesting = interesting
        self.update_relus()
        

    def update_relus(self):
        def hook(module, grad_in, grad_out):
            self.grads.insert(0,grad_out[0])

        # Loop through layers, hook up ReLUs
        for i in range(len(self.model.features)):
            self.handles.append(self.model.features[i].register_backward_hook(hook))

    def release(self):
        super(LayerWise_VG,self).release()
        self.forward_relu_outputs = []


    def interpret(self, x):
        x.requires_grad=True
        ret = []
        with torch.enable_grad():
            self.model_output,layer_output,features = self.model.forward_layer(x)
            self.model.zero_grad()
            self.model_pred = self.model_output.max(1)[1]
            y_onehot = self.to_one_hot(self.model_pred, self.model_output.shape[1]).float()
            loss = (y_onehot * self.model_output).sum()                 # 用logit回传梯度
            #loss = F.cross_entropy(self.model_output, self.model_pred) # 计算Loss用的标签是模型输出标签
            loss.backward(retain_graph=True)
            for i in self.interesting:
                ret.append(self.grads[i].data)
            #print(x.grad.data[0,1])
            return x.grad.data,ret