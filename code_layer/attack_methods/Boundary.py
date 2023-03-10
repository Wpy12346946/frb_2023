import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
import foolbox
import torch
import numpy as np
# from torch.autograd.gradcheck import zero_gradients
from advertorch.attacks import CarliniWagnerL2Attack, LinfPGDAttack
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# attack=foolbox.attacks.BoundaryAttack()
# def Boundray_attack(opt,data,model,labels):
#     targeted = opt.attack.endswith('T')
#     # data = data*255.0
#     device=opt.device
#     fmodel=foolbox.models.PyTorchModel(model,bounds=(0,1),device=device)
#     if targeted:
#         criterion = foolbox.criteria.TargetedMisclassification(labels)
#     else:
#         criterion = foolbox.criteria.Misclassification(labels)
#     image=attack(fmodel,data,criterion=criterion,epsilons=0.5)
#     # print(image)
#     # image=image[0]/255.0
#     image=image[0]
#     return image

class Boundary_attacker:
    def __init__(self,opt,model):
        self.attacker=foolbox.attacks.BoundaryAttack()
        self.fmodel=foolbox.models.PyTorchModel(model,bounds=(0,1),device=opt.device)
        self.targeted=opt.attack.endswith('T')
        self.device=opt.device
        # self.eps=eps
    def __call__(self,org,org_l,pair,pair_l):
        if self.targeted:
            criterion = foolbox.criteria.TargetedMisclassification(pair_l)
        else:
            criterion = foolbox.criteria.Misclassification(org_l)
        ret=self.attacker(self.fmodel,org,criterion=criterion,epsilons=0.5,starting_points=pair)
        image=ret[0]
        return image