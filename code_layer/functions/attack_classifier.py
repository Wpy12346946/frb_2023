import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from models import *
from datasets import build_loader
from attack_methods import adversarialattack
import shutil
from random import randint

def attack(opt):
    clean_data_root = os.path.join(opt.adversarial_root, '{}_{}_Clean'.format(opt.dataset, opt.classifier_net))
    # Check the maps root
    if os.path.exists(clean_data_root):
        print("({}) clean data are already saved. Now to generate the adversarial data...".format(opt.attack))
        generateclean = False
    else:
        print("Now saving clean data of {} ...".format(opt.attack))
        generateclean = True

    adver_data_root = os.path.join(opt.adversarial_root, '{}_{}_{}'.format(opt.dataset, opt.classifier_net, opt.attack))
    # Check the maps root
    if os.path.exists(adver_data_root):
        chars = input("\nThe TVM data file of this exp already exists "
                      "and it may be overlapped by the new ones. Continued? \n"
                      "yes: reomve the old one and generate a new one \n"
                      "no:  return\n"
                      "[yes/no]: ")
        if chars in 'yes':
            print("Removing......")
            shutil.rmtree(adver_data_root)
            os.makedirs(adver_data_root)
            print("Done!")
        elif chars in 'no':
            return 
            # adver_data_root = os.path.join(opt.adversarial_root, '{}_{}_{}_{}'\
                                     # .format(opt.exp_name,opt.dataset, opt.classifier_net,  opt.attack))
            # os.makedirs(adver_data_root)
        else:
            raise Exception("Wrong input!!!")

    # Load classifier
    classifier = load_classifier(opt)

    dataloaders = {}
    dataloaders['train'] = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=True, workers=opt.workers)
    dataloaders['test'] = build_loader(opt.data_root, opt.dataset, opt.val_batchsize, train=False, workers=opt.workers)

    if generateclean:
        choose(clean_data_root, classifier, dataloaders, opt, justclean=True)
    choose(adver_data_root, classifier, dataloaders, opt, justclean=False)

def choose(root, classifier, dataloaders, opt, justclean):
    for phase in ['test', 'train']:
        #root example '../adversarial_data/cifar10_vgg11bn_FGSM-U/train/
        root_ = root + '/{}/'.format(phase)
        if not os.path.exists(root_):
            os.makedirs(root_)

        print("Size of the {} set of {} is: {}".format(phase, opt.dataset, len(dataloaders[phase].dataset)))

        # FGSM attack uses multiple epsilons, here to limit the dataset size.
        if opt.attack.startswith("FGSM") and not justclean:
            maxdatasize = {'test': opt.fgsm_test_max, 'train': opt.fgsm_train_max}
            for fgsmeps in opt.fgsm_epsilons:
                generate_adver_data(classifier, dataloaders[phase], fgsmeps, root_, opt, justclean, maxdatasize[phase])

        # Other attacks don't require multiple parameter epsilons, so it is set to 0.
        else:
            generate_adver_data(classifier, dataloaders[phase], 0, root_, opt, justclean)

def generate_adver_data(classifier, data_loader, eps, root, opt, justclean=True, maxdatasize=None):
    device = opt.device
    correct = 0
    saved_cnt = 0
    num_wrong = 0
    datasize = 0

    for data, label in tqdm(data_loader, desc=opt.attack if not justclean else "Clean"):
        datasize += data.size(0)
        data, label = data.to(device), label.to(device)
        data.requires_grad = True  # Set requires_grad attribute of tensor. Important for attack method

        # Record classify wrong indexes
        clean_pred = classifier(data).max(1)[1]
        classify_wrong_index = (clean_pred - label).nonzero()[:, 0]
        num_wrong += classify_wrong_index.size(0)

        targeted = opt.attack.endswith('T')
        if targeted:
            targeted_label = []
            # Randomly choose targeted label that is not the ground truth one
            for i in range(data.size(0)):
                targeted_label.append(randint(1, opt.classifier_classes-1))
            attack_label = torch.fmod(label + torch.tensor(targeted_label).long().to(device), opt.classifier_classes)
        else:
            # Untargeted attacks use the model classification labels
            attack_label = clean_pred

        if not justclean:
            perturbed_data = adversarialattack(opt.attack, classifier, data, attack_label, eps, opt)
            perturbed_data.requires_grad = True
            adver_pred = classifier(perturbed_data).max(1)[1]

            # Record attack success indexes; remove classifiy_wrong_index
            if targeted:
                attack_success_index = (adver_pred == attack_label).nonzero()[:, 0]
            else:
                attack_success_index = (adver_pred != attack_label).nonzero()[:, 0]
            attack_success_index = list(attack_success_index)
            for i in list(classify_wrong_index):
                if i in attack_success_index:
                    attack_success_index.remove(i)
            correct += data.size(0) - classify_wrong_index.size(0) - len(attack_success_index)

            # When the attack succeeds, store the adversarial data
            for index in attack_success_index:
                index = index.item()

                torch.save(perturbed_data[index].detach().cpu(),
                           '{}/image_E{}_N{}_T{}_F{}.pth' \
                           .format(root, eps, saved_cnt, clean_pred[index].item(), adver_pred[index].item()))
                saved_cnt += 1

            if (opt.attack.startswith('FGSM')) and saved_cnt >= maxdatasize:
                break

        else:
            classify_right_index = list(range(data.size(0)))
            for i in list(classify_wrong_index):
                classify_right_index.remove(i)

            correct += len(classify_right_index)

            # When the classifier predicts rightly, store the maps of clean
            for index in classify_right_index:
                torch.save(data[index].detach().cpu(),
                           '{}/image_E{}_N{}_T{}.pth' \
                           .format(root, eps, saved_cnt, clean_pred[index].item()))
                saved_cnt += 1


    if justclean:
        print("\nEpsilon: {}\tModel Accuracy = {} / {} = {:.2f}%" \
              .format(eps, correct, datasize, (correct/datasize)*100))
        print("Clean data set size: ", correct, '\n')

    else:
        final_acc = correct / float(datasize - num_wrong)
        print("\nEpsilon: {}\tModel Accuracy Under Attack = {} / {} = {:.2f}%"\
              .format(eps, correct, datasize - num_wrong, final_acc*100))
        print("Adversarial Attack Rate = {:.2f}%".format((1-final_acc)*100))
        print("Adver data set size: ", datasize - num_wrong - correct,'\n')
