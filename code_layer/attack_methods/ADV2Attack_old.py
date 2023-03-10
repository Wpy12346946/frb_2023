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


from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import is_successful
from advertorch.attacks.iterative_projected_gradient import PGDAttack
import torch.nn.functional as F

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

class ADV2Attack(Attack, LabelMixin):
    """
    The ADV^2 Attack of BPG Interpretation based, Interpretable Deep Learning under Fire 

    :param predict: forward pass function.
    :param num_classes: number of clasess.
    :param confidence: confidence of the TVM examples.
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
            self, predict, num_classes, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1., lambda_int = 0.007,
            ord=np.inf, l1_sparsity=None, targeted=False):
        """Carlini Wagner L2 Attack implementation in pytorch."""
        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently do not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

            loss_fn = None

        super(ADV2Attack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l1_sparsity = l1_sparsity
        self.num_classes = num_classes
        self.lambda_int = lambda_int
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)
    
    def _loss_fn(self, output, y_onehot, l2distsq, const):
        f_ct = (y_onehot * output).sum(dim=1)
        
        loss1 = -torch.log(f_ct)
        loss2 = const * l2distsq
        
        # loss = (loss1 + loss2).sum()
        loss = (loss1 + loss2).mean()
        
        return loss

    def get_interpret(self, x):
        image = Variable(x.data, requires_grad = True)
        # x.requires_grad = True
        model_output = self.predict(image)
        self.predict.zero_grad()
        model_pred = model_output.max(1)[1]
        loss = F.cross_entropy(model_output, model_pred) # 计算Loss用的标签是模型输出标签
        loss.backward(retain_graph=True)
        gradients = image.grad.data
        self.predict.zero_grad()
        return gradients

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their TVM counterparts with
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
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self._loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity,
        )

        return rval.data

    def perturb_iterative(self, xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0,
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

        y_onehot = to_one_hot(yvar, self.num_classes).float()
        m_t = self.get_interpret(xvar)

        for ii in range(nb_iter):
            outputs = predict(xvar + delta)
            g_x_f = self.get_interpret(xvar + delta)
            l2distsq = calc_l2distsq(g_x_f, m_t)
            loss = self._loss_fn(outputs, y_onehot, l2distsq, self.lambda_int)

            loss.backward()
            if ord == np.inf:
                grad_sign = delta.grad.data.sign()
                delta.data = delta.data - batch_multiply(eps_iter, grad_sign)   # adv2 update 
                delta.data = batch_clamp(eps, delta.data)
                delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                ) - xvar.data
            else:
                error = "Only ord = inf have been implemented"
                raise NotImplementedError(error)
            delta.grad.data.zero_()

        x_adv = clamp(xvar + delta, clip_min, clip_max)
        return x_adv







