import torch
import torch.nn as nn
from .InterpreterBase import Interpreter

class VanilaGradients(Interpreter):
    def __init__(self, model):
        super(VanilaGradients, self).__init__(model)