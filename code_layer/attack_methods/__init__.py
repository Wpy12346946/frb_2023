from physical_attack_method.roa import ROA
from .attacks import *
import torch
from .Boundary import Boundary_attacker
from .NES import NES_attacker
from .Optim import Optim_attacker
from .TwoLossPGD import TowLoss_attacker
from .TwoLossWhiteBox import TowLossWhiteBox_attacker
from .PatchAttack import PatchAttacker

attackmethods = ['FGSM-U', 'PGD-U', 'PGD-T', 'DFool-U', 'CW-U', 'CW-T', 'DDN-U', 'DDN-T', 'ADV2-T']


def adversarialattack(attackmethod: str, model, data, attack_labels, opt, eps=0.01):
    targeted = attackmethod.endswith('T')

    if attackmethod.startswith('FGSM'):
        if hasattr(opt, 'fgsm_eps'):
            eps = opt.fgsm_eps
        perturbed_data = fgsm_attack(model, data, attack_labels, eps, targeted=targeted)

    # DeepFool attack is only untargeted
    elif attackmethod.startswith('DFool'):
        perturbed_data = deepfool_attack(model, data, attack_labels, num_classes=opt.classifier_classes,
                                         overshoot=opt.dfool_overshoot, max_iter=opt.dfool_max_iterations)

    elif attackmethod.startswith('CW'):
        perturbed_data = cw_attack(model, data, attack_labels, num_classes=opt.classifier_classes,
                                   targeted=targeted, learning_rate=opt.cw_lr, max_iterations=opt.cw_max_iterations,
                                   confidence=opt.cw_confidence)

    elif attackmethod.startswith('DDN'):
        perturbed_data = ddn_attack(model, data, attack_labels, istargeted=targeted, steps=opt.dnn_steps)

    elif attackmethod.startswith('PGD'):
        perturbed_data = pgd_attack(model, data, attack_labels, istargeted=targeted,
                                    eps=opt.pgd_eps, nb_iter=opt.pgd_iterations, eps_iter=opt.pgd_eps_iter)
    elif attackmethod.startswith('ADV2'):
        perturbed_data = adv2_attack(model, data, attack_labels, opt.classifier_classes, targeted,
                                     eps=opt.pgd_eps, nb_iter=opt.pgd_iterations, eps_iter=opt.pgd_eps_iter)
    elif attackmethod.startswith('ROA'):
        roa = ROA(model, data.size(2))
        learning_rate = 0.1
        iterations = 5
        ROAwidth = 5
        ROAheight = 5
        skip_in_x = 2
        skip_in_y = 2
        potential_nums = 5
        perturbed_data = roa.gradient_based_search(data, attack_labels, learning_rate, iterations,
                                                   ROAwidth, ROAheight, skip_in_x, skip_in_y,
                                                   potential_nums)
        return perturbed_data
    else:
        raise Exception("No such attack method {}".format(attackmethod))

    perturbed_data = perturbed_data.detach()
    # Adding clipping to maintain range of [0,1]
    perturbed_data = torch.clamp(perturbed_data, 0.0, 1.0)
    return perturbed_data


def blackboxAttacker(attackmethod: str, model, opt):
    if attackmethod.startswith('Boundary'):
        attacker = Boundary_attacker(opt, model)
    elif attackmethod.startswith('NES'):
        attacker = NES_attacker(opt, model)
    elif attackmethod.startswith('Optim'):
        attacker = Optim_attacker(opt, model)
    else:
        raise Exception("No such attack method {}".format(attackmethod))

    return attacker
