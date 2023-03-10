import torch
import numpy as np
# from torch.autograd.gradcheck import zero_gradients
from advertorch.attacks import CarliniWagnerL2Attack, LinfPGDAttack
# from .ADV2Attack_old import ADV2Attack
from .ADV2Attack import ADV2Attack
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__all__ = ['fgsm_attack', 'bim_attack', 'cw_attack',  'deepfool_attack', 'ddn_attack', 'pgd_attack', 'adv2_attack']

def fgsm_attack(model, data, label, epsilon, targeted=False):
    model.zero_grad()
    model_output = model(data)

    loss = F.cross_entropy(model_output, label)
    loss.backward()
    data_grad = data.grad.data.clone().detach()
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    if targeted:
        # targeted attack minimizes the targeted label loss
        perturbed_image = data - epsilon * sign_data_grad
    else:
        # untargeted attack maximizes the groud truth label loss
        perturbed_image = data + epsilon * sign_data_grad
    
    model.zero_grad()

    return perturbed_image.clone().detach()

def bim_attack(model, data, label, epsilon, alpha, iteration, targeted=False):
    perturbed_image = data.detach().clone()

    # Start iteration
    for i in range(iteration):
        perturbed_image.requires_grad = True

        model.zero_grad()
        model_output = model(perturbed_image)
        loss = F.cross_entropy(model_output, label)
        loss.backward()

        data_grad = perturbed_image.grad.data
        adv_noise = alpha * data_grad.sign()
        # clipped_adv_noise = torch.clamp(adv_noise, -epsilon, epsilon)

        if targeted:
            perturbed_image = data + torch.clamp(perturbed_image - adv_noise - data, -epsilon, epsilon)
        else:
            perturbed_image = data + torch.clamp(perturbed_image + adv_noise - data, -epsilon, epsilon)
        perturbed_image = perturbed_image.detach()

    return perturbed_image

def cw_attack(model, data, label, num_classes, targeted=False, learning_rate=0.01, max_iterations=10000, confidence=0):
    adversary = CarliniWagnerL2Attack(model, num_classes, confidence=confidence, targeted=targeted,
                                      learning_rate=learning_rate, binary_search_steps=9, max_iterations=max_iterations,
                                      abort_early=True, initial_const=0.001, clip_min=0.0, clip_max=1.0,
                                      loss_fn=None)
    cln_data = data
    true_label = label
    adv_untargeted = adversary.perturb(cln_data, true_label)
    return adv_untargeted

def ddn_attack(model, data, label, istargeted = False, steps=1000):
    adversary = DDN(steps = steps, quantize = False, max_norm = 0.5, device = data.device)

    cln_data = data
    true_label = label
    adv_samples = adversary.attack(model, cln_data, labels=true_label, targeted=istargeted)
    return adv_samples

def deepfool_attack(model, data, label, num_classes, overshoot=0.02, max_iter=50):
    cln_data = data
    true_label = label
    # no need to keep the first four return values
    # please refer to the deepfool method details below for more on these ignored values
    _, _, _, _, adv_untargeted = deepfool(cln_data, model, num_classes, overshoot, max_iter)
    return adv_untargeted

def pgd_attack(model, data, label, istargeted = False, eps=0.3, nb_iter=40, eps_iter= 0.01):
    adversary = LinfPGDAttack(model, loss_fn=None, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=True, clip_min=0., clip_max=1.,
            targeted=istargeted)

    cln_data = data
    true_label = label
    adv_examples = adversary.perturb(cln_data, true_label)
    return adv_examples



def deepfool(batch_img, model, num_classes, overshoot, max_iter):
    """
    Parameters
    ----------
    batch_img: torch.(cuda.)Tensor
        Image batch of shape batchsize x (HxWx3)
    model: nn.Module
        network (input: images, output: values of activation **BEFORE** softmax).
    num_classes: int
        num_classes (limits the number of classes to test against, by default = 10)
    overshoot: float
        used as a termination criterion to prevent vanishing updates (default = 0.02).
    max_iter: int
        maximum number of iterations for deepfool (default = 50)
    Returns
    -------
    r_tot: int
        minimal perturbation that fools the classifier
    loop_i: int
        number of iterations that finding a batch of perturbed image requires
    label: torch.tensor
        raw label
    k_i: torch.tensor
        new estimated_label
    pert_image: torch.tensor
        perturbed image
    """
    # get the default device prepared for other newly created tensor
    device = batch_img.device
    batch_img.requires_grad = False
    # get the index of the max log-probability
    label = model(batch_img).max(1)[1]

    input_shape = batch_img.shape #torch.Size
    pert_image = batch_img.clone()#pert image initialize
    w = torch.zeros(input_shape).to(device) #linearized approximation of boundary
    r_tot = torch.zeros(input_shape).to(device)  #noise variable initialize
    pert = (torch.ones_like(label, dtype=torch.float32).reshape(-1,1).to(device)*np.inf)

    pert_image.requires_grad = True
    fake_score = model(pert_image)
    k_i = label #pert image predicted label initialize
    loop_i = 0
    is_success = torch.zeros(input_shape[0]).reshape([-1, 1]).to(device)
    while torch.sum(is_success) < input_shape[0] and loop_i < max_iter:
        #the predicted label's score
        score_orig = fake_score.gather(1, label.reshape(-1, 1).long())
        #the predicted label's backprop with shape [batchsize , 1]
        score_orig.backward(torch.ones_like(label,dtype=torch.float32).reshape([-1,1]).to(device), retain_graph=True)
        #original grad of input x
        grad_orig = pert_image.grad.data.clone()

        for k in range(0, num_classes):
            # variable version of zero grad
            # And definitely a legacy version which does not even occur in 1.0.1...
            # zero_gradients(pert_image)
            if pert_image.grad is not None:
                pert_image.grad.zero_()
            # backward k_th fake score with shape [batchsize, ]
            fake_score[:, k].backward(torch.ones_like(label,dtype=torch.float32).to(device),retain_graph=True)
            # current kth grad
            cur_grad = pert_image.grad.data

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig #shape == (batch x 3xHxW)
            f_k = (fake_score[:, k].reshape([-1,1]) - score_orig) # shape == batch x 1
            # per batch line norm, per batch line division, per batch line flatten
            # pert_k shape batch x 1
            pert_k = torch.div(torch.abs(f_k), torch.norm(w_k.flatten(start_dim=1), dim=1, keepdim=True))

            for i, tmp_pert_k in enumerate(pert_k):
                if label[i] == k:
                    # discard the same class with true label
                    continue
                # determine which w_k to use
                if tmp_pert_k < pert[i]:
                    pert[i] = tmp_pert_k.data
                    w[i] = w_k[i].data

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        # noise = pert amount * normalized weight
        # wieght: the normal vector of the estimated linearized boundary
        r_i =  (pert+1e-4) * torch.div(w.flatten(start_dim=1), torch.norm(w.flatten(start_dim=1), dim=1, keepdim=True))
        # set already succeed batch line's noise r_i to zero
        r_i = r_i - torch.mul(r_i, is_success)
        r_tot = torch.add(r_tot, r_i.reshape(r_tot.shape))

        pert_image = batch_img + (1+overshoot)*r_tot.to(batch_img.device)
        pert_image = torch.clamp(pert_image, 0.0, 1.0)
        pert_image.requires_grad = True


        fake_score = model(pert_image)
        k_i = fake_score.max(1)[1]
        # update issuccess recording successfully attacked batch line
        is_success = (k_i != label).type(torch.float32).reshape([-1, 1])
        loop_i += 1

    r_tot = (1+overshoot)*r_tot
    return r_tot, loop_i, label, k_i, pert_image


def adv2_attack(model, data, label, num_classes, istargeted=True, eps=0.031, nb_iter=200, eps_iter=0.0039, lambda_int=0.007,
                warm_start=True):
    adversary = ADV2Attack(model, num_classes, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, lambda_int=lambda_int, targeted=istargeted)

    cln_data = data
    target_label = label
    if warm_start:
        adversary2 = LinfPGDAttack(model, loss_fn=None, eps=eps, nb_iter=nb_iter,
                                   eps_iter=eps_iter, rand_init=True, clip_min=0., clip_max=1.,
                                   targeted=istargeted)

        cln_data = adversary2.perturb(cln_data, target_label)
        adv_examples = adversary2.perturb(cln_data, target_label)

    adv_examples = adversary.perturb(cln_data, target_label)
    return adv_examples


class DDN:
    """
    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    gamma : float, optional
        Factor by which the norm will be modified. new_norm = norm * (1 + or - gamma).
    init_norm : float, optional
        Initial value for the norm.
    quantize : bool, optional
        If True, the returned adversarials will have quantized values to the specified number of levels.
    levels : int, optional
        Number of levels to use for quantization (e.g. 256 for 8 bit images).
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.
    callback : object, optional
        Visdom callback to display various metrics.
    """

    def __init__(self,
                 steps: int,
                 gamma: float = 0.05,
                 init_norm: float = 1.,
                 quantize: bool = True,
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu'),
                 callback: Optional = None) -> None:
        self.steps = steps
        self.gamma = gamma
        self.init_norm = init_norm

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm

        self.device = device
        self.callback = callback

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               targeted: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.
        Returns
        -------
        (inputs + best_delta): torch.Tensor
            Batch of samples modified to be TVM to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)
        norm = torch.full((batch_size,), self.init_norm, device=self.device, dtype=torch.float)
        worst_norm = torch.max(inputs, 1 - inputs).view(batch_size, -1).norm(p=2, dim=1)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps, eta_min=0.01)

        best_l2 = worst_norm.clone()
        best_delta = torch.zeros_like(inputs)
        adv_found = torch.zeros(inputs.size(0), dtype=torch.uint8, device=self.device)

        for i in range(self.steps):
            scheduler.step()

            l2 = delta.data.view(batch_size, -1).norm(p=2, dim=1)
            adv = inputs + delta
            logits = model(adv)
            pred_labels = logits.argmax(1)
            ce_loss = F.cross_entropy(logits, labels, reduction='sum')
            loss = multiplier * ce_loss

            is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
            is_smaller = l2 < best_l2
            is_both = is_adv * is_smaller
            adv_found[is_both] = 1
            best_l2[is_both] = l2[is_both]
            best_delta[is_both] = delta.data[is_both]

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            if self.callback:
                cosine = F.cosine_similarity(-delta.grad.view(batch_size, -1),
                                             delta.data.view(batch_size, -1), dim=1).mean().item()
                self.callback.scalar('ce', i, ce_loss.item() / batch_size)
                self.callback.scalars(
                    ['max_norm', 'l2', 'best_l2'], i,
                    [norm.mean().item(), l2.mean().item(),
                     best_l2[adv_found].mean().item() if adv_found.any() else norm.mean().item()]
                )
                self.callback.scalars(['cosine', 'lr', 'success'], i,
                                      [cosine, optimizer.param_groups[0]['lr'], adv_found.float().mean().item()])

            optimizer.step()

            norm.mul_(1 - (2 * is_adv.float() - 1) * self.gamma)
            norm = torch.min(norm, worst_norm)

            delta.data.mul_((norm / delta.data.view(batch_size, -1).norm(2, 1)).view(-1, 1, 1, 1))
            delta.data.add_(inputs)
            if self.quantize:
                delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)
            delta.data.clamp_(0, 1).sub_(inputs)

        if self.max_norm:
            best_delta.renorm_(p=2, dim=0, maxnorm=self.max_norm)
            if self.quantize:
                best_delta.mul_(self.levels - 1).round_().div_(self.levels - 1)

        return inputs + best_delta
