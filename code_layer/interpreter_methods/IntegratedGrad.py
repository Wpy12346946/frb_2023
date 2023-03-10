import torch
import torch.nn as nn
import torch.nn.functional as F
from .InterpreterBase import Interpreter
import numpy as np
import time
from memory_profiler import profile

class IntegratedGrad(Interpreter):
    def __init__(self, model):
        super(IntegratedGrad, self).__init__(model)
        self.steps=5

    def predict_gradients(self,inputs):
        outputs=[]
        # print(len(inputs))
        for ind,x in enumerate(inputs):
            x.detach_()
            # x=x.detach().clone()
            x.requires_grad=True
            model_output = self.model(x)
            self.model.zero_grad()

            y_onehot = self.to_one_hot(self.model_pred, model_output.shape[1]).float()
            y_onehot.detach_()
            loss = (y_onehot * model_output).sum()  # ��logit�ش��ݶ�
            # loss = F.cross_entropy(model_output, model_pred) # ����Loss�õı�ǩ��ģ�������ǩ
            loss.backward(retain_graph=True)
            outputs.append(x.grad.data)
        output_avg=0
        for x in outputs:
            output_avg=output_avg+x
        output_avg=output_avg/len(outputs)
        return output_avg

    def interpret(self, x):
        baseline=0*x
        # baseline=np.random.random(x.shape)
        self.model_output = self.model(x)
        self.model.zero_grad()
        self.model_pred = self.model_output.max(1)[1]
        steps = self.steps

        scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(0, steps + 1)]
        t0=time.time()
        grad = self.predict_gradients(scaled_inputs)
        self.model.zero_grad()
        t1=time.time()
        # print("{:.4f}".format(t1-t0))
        integrated_grad = (x - baseline) * grad
        return integrated_grad
