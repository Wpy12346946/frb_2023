import torch
import torch.nn as nn
from .InterpreterBase import Interpreter
from .LRP_utils.innvestigator import InnvestigateModel

class LRP(Interpreter):
    def __init__(self, model):
        super(LRP, self).__init__(model)
        self.model=model
        

    def interpret(self,x):
        device=x.device
        self.inn_model = InnvestigateModel(self.model,device=device, lrp_exponent=2,
                              method="e-rule",
                              beta=.5)
        self.handles=self.inn_model.get_handles()
        model_prediction=self.inn_model.evaluate(x)
        self.model_output=model_prediction
        self.model_pred = self.model_output.max(1, keepdim=True)[1].squeeze(1)
        _, true_relevance = self.inn_model.innvestigate()
        self.data_grad=true_relevance.to(device)
        return self.data_grad
