import torch
import torch.nn as nn
# import numpy as np
# from torch.autograd import Variable
from .InterpreterBase import Interpreter


class SmoothGrad(Interpreter):
    def __init__(self, model):
        super(SmoothGrad, self).__init__(model)
        self.stdev_spread = 0.15 #参数
        self.n_samples = 25 #参数

    def interpret(self, x):
        x = x.detach()
        stdev = self.stdev_spread * (torch.max(x) - torch.min(x)) #* (np.max(x) - np.min(x)) #标准偏差的无偏极差估计
        total_gradients = torch.zeros_like(x)
        for i in range(self.n_samples):
            noise = torch.normal(torch.zeros_like(x), stdev * torch.ones_like(x))
            x_plus_noise = x + noise
            x_plus_noise.requires_grad=True
            model_output = self.model(x_plus_noise)
            self.model_pred = model_output.max(1)[1]
            one_hot = self.to_one_hot(self.model_pred, model_output.shape[1]).float()
            loss = (one_hot * model_output).sum()  # 用logit回传梯度
            self.model.zero_grad()

            loss.backward(retain_graph=True)
            grad = x_plus_noise.grad.data

            total_gradients += grad

        avg_gradients = total_gradients / self.n_samples

        return  avg_gradients

