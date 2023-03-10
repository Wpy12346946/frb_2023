from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from advertorch.utils import calc_l2distsq
from advertorch.utils import tanh_rescale
from advertorch.utils import torch_arctanh
from advertorch.utils import clamp
from advertorch.utils import to_one_hot
from advertorch.utils import replicate_input
from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input
from advertorch.utils import batch_l1_proj

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import is_successful
from advertorch.attacks.iterative_projected_gradient import PGDAttack
import torch.nn.functional as F
import time
from tqdm import tqdm

# from advertorch.attacks.iterative_projected_gradient.utils import rand_init_delta

def rand_init_delta(delta, x, ord, eps, clip_min, clip_max):
    # TODO: Currently only considered one way of "uniform" sampling
    # for Linf, there are 3 ways:
    #   1) true uniform sampling by first calculate the rectangle then sample
    #   2) uniform in eps box then truncate using data domain (implemented)
    #   3) uniform sample in data domain then truncate with eps box
    # for L2, true uniform sampling is hard, since it requires uniform sampling
    #   inside a intersection of cube and ball, so there are 2 ways:
    #   1) uniform sample in the data domain, then truncate using the L2 ball
    #       (implemented)
    #   2) uniform sample in the L2 ball, then truncate using the data domain
    # for L1: uniform l1 ball init, then truncate using the data domain

    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.data.uniform_(-1, 1)
        delta.data = batch_multiply(eps, delta.data)
    else:
        error = "Only ord = inf have been implemented"
        raise NotImplementedError(error)

    delta.data = clamp(
        x + delta.data, min=clip_min, max=clip_max) - x
    return delta.data

class TowLossAttack(Attack, LabelMixin):
    """
    The ADV^2 Attack of BPG Interpretation based, Interpretable Deep Learning under Fire 

    :param predict: forward pass function.
    :param confidence: confidence of the adversarial examples.
    :param targeted: if the attack is targeted.
    :param learning_rate: the learning rate for the attack algorithm
    :param binary_search_steps: number of binary search times to find the
        optimum
    :param max_iterations: the maximum number of iterations
    :param abort_early: if set to true, abort early if getting stuck in local
        min
    :param initial_const: initial value of the constant c
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param loss_fn: loss function
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, l1_sparsity=None, targeted=False,sample_batch_size=50):
        """Carlini Wagner L2 Attack implementation in pytorch."""
        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently do not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

            loss_fn = None

        super(TowLossAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn  = nn.CrossEntropyLoss(reduction="sum")
            self.loss_fn2 = nn.CrossEntropyLoss(reduction='sum')
        self.l1_sparsity = l1_sparsity
        self.sample_batch_size=sample_batch_size
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)
    

    def perturb(self, x, y, cls_y,choose):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)
        
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x
        
        rval = self.perturb_iterative(
            x, y, cls_y, self.predict,choose, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity,
        )

        return rval.data

    def perturb_iterative(self, xvar, yvar, cls_y, predict,choose, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0,momentum=0.1,
                      l1_sparsity=None):
        """
        Iteratively maximize the loss over the input. It is a shared method for
        iterative attacks including IterativeGradientSign, LinfPGD, etc.

        :param xvar: input data.
        :param yvar: input labels.
        :param predict: forward pass function.
        :param nb_iter: number of iterations.
        :param eps: maximum distortion.
        :param eps_iter: attack step size.
        :param loss_fn: loss function.
        :param delta_init: (optional) tensor contains the random initialization.
        :param minimize: (optional bool) whether to minimize or maximize the loss.
        :param ord: (optional) the order of maximum distortion (inf or 2).
        :param clip_min: mininum value per input dimension.
        :param clip_max: maximum value per input dimension.
        :param l1_sparsity: sparsity value for L1 projection.
                    - if None, then perform regular L1 projection.
                    - if float value, then perform sparse L1 descent from
                        Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
        :return: tensor containing the perturbed input.
        """
        if delta_init is not None:
            delta = delta_init
        else:
            delta = torch.zeros_like(xvar)

        delta.requires_grad_()

        grad_prev = torch.zeros_like(xvar,device=xvar.device)
        # all_acc=[]
        # choose_sz=torch.sum(choose).detach().item()

        for ii in range(nb_iter):
            # print(f'{ii}/{nb_iter}')
            if ord == np.inf:
                outputs = predict(xvar + delta)
                classifier_output = predict.classifier_output
                # loss_fn is targeted , loss_fn2 is untargeted
                # loss = self.loss_fn(outputs, yvar) - self.loss_fn2(classifier_output,cls_y)
                loss = self.loss_fn(outputs, yvar)
                loss.backward()
                grad = delta.grad

                # if minimize:
                    # grad=-grad
                grad = momentum*grad_prev + (1-momentum)*grad
                grad_prev=grad
                grad_sign = grad.data.sign()
                delta.data = delta.data + batch_multiply(eps_iter, grad_sign)   # pgd update 
                delta.data = batch_clamp(eps, delta.data)
                delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                ) - xvar.data
            else:
                error = "Only ord = inf have been implemented"
                raise NotImplementedError(error)
                # delta.grad.data.zero_()
            out_labels=predict(xvar+delta).max(1)[1]
            if minimize:
                acc=torch.sum(out_labels[choose]==yvar[choose]).detach().item()
            else:
                acc=torch.sum(out_labels[choose]!=yvar[choose]).detach().item()
            batch_size=xvar.shape[0]
            # print('current acc={}/{}={:.4f}'.format(acc,choose_sz,acc/choose_sz))
            # all_acc.append(acc/choose_sz)
            # print(out_labels,yvar)
            # update= ii % 10 ==0
            # if minimize:
            #     print(torch.sum(out_labels==yvar))
            #     if torch.sum(out_labels==yvar)==batch_size:
            #         update=True
            # else:
            #     print(torch.sum(out_labels!=yvar))
            #     if torch.sum(out_labels!=yvar)==batch_size:
            #         update=True
            # if update:
            #     eps =eps*0.9
            #     eps = 0.1 if eps<0.1 else eps
            #     eps_iter=eps_iter*0.9
            #     eps_iter=0.001 if eps_iter<0.001 else eps_iter
            torch.cuda.empty_cache()


        x_adv = clamp(xvar + delta, clip_min, clip_max)
        # self.all_acc=all_acc
        return x_adv



class TowLoss_attacker:
    def __init__(self,opt,model):
        self.sample_batch_size=50
        self.eps=opt.pgd_eps
        self.nb_iter=opt.pgd_iterations
        self.eps_iter=opt.pgd_eps_iter
        istargeted=opt.attack.endswith('T')
        self.istargeted=istargeted
        self.model=model
        self.adversary = TowLossAttack(model, eps=self.eps, nb_iter=self.nb_iter,
         eps_iter=self.eps_iter, targeted=istargeted,sample_batch_size=self.sample_batch_size)
        # print(f'using attacker:TowLoss  params:eps={self.eps},lr={self.eps_iter},nb_iter={self.nb_iter},istargeted={istargeted}')

    def __call__(self,x,y,cls_y):
        choose=self.model(x).max(1)[1]==y

        image=self.adversary.perturb(x, y, cls_y, choose)
        return image