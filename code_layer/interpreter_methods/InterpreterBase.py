import torch
import torch.nn as nn
import torch.nn.functional as F
# from advertorch.utils import to_one_hot

def to_one_hot(y, num_classes=10):
    device=y.device
    batch_size=y.shape[0]
    onehot= torch.zeros(batch_size,num_classes,device=device).scatter_(1,y.unsqueeze(1),1)
    return onehot

class Interpreter():
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.model_output = None
        self.model_pred = None
        self.to_one_hot = to_one_hot
        self.model.eval()

    def interpret(self, x, pred=None):
        x.requires_grad=True
        with torch.enable_grad():
            self.model_output = self.model(x)
            self.model.zero_grad()
            self.model_pred = self.model_output.max(1)[1]
            if pred is None:
                y_onehot = self.to_one_hot(self.model_pred, self.model_output.shape[1]).float()
            else:
                y_onehot = self.to_one_hot(pred, self.model_output.shape[1]).float()
            loss = (y_onehot * self.model_output).sum()                 # 用logit回传梯度
            #loss = F.cross_entropy(self.model_output, self.model_pred) # 计算Loss用的标签是模型输出标签
            loss.backward(retain_graph=True)
            #print(x.grad.data[0,1])
            return x.grad.data

    # def interpret(self, x, pred=None):
    #     x.requires_grad=True
    #     loss_fn = nn.CrossEntropyLoss(reduction="sum")
    #     with torch.enable_grad():
    #         self.model_output = self.model(x)
    #         self.model.zero_grad()
    #         self.model_pred = self.model_output.max(1)[1]
    #         if pred is None:
    #             loss = loss_fn(self.model_output,self.model_pred)
    #         else:
    #             loss = loss_fn(self.model_output,pred)
    #         loss.backward(retain_graph=True)
    #         #print(x.grad.data[0,1])
    #         return x.grad.data

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
