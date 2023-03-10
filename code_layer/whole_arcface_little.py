import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from Options import Options, report_args
import os
from datasets import build_loader, pairs_loader
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datasets import build_maps_loader
from models import *
import numpy as np
from sklearn.metrics import roc_auc_score
import torchvision.utils as vutils
import pandas as pd
from datetime import datetime
import random
from random import randint
from attack_methods import adversarialattack, blackboxAttacker
import re, math
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from interpreter_methods import interpretermethod
from detectors.XEnsemble import *
from code_layer.face.test import arcfacebn, predict_batch

warnings.filterwarnings("ignore")

import pickle


def pk_dump(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def pk_load(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


@torch.enable_grad()
def getRec(classifier, data, label):
    data = data.detach().clone()
    data.requires_grad = True

    clean_pred = classifier(data).max(1)[1]
    attack = opt.attack
    cw_max_iterations = opt.cw_max_iterations
    cw_confidence = opt.cw_confidence
    pgd_eps = opt.pgd_eps
    pgd_eps_iter = opt.pgd_eps_iter
    pgd_iterations = opt.pgd_iterations
    opt.attack = opt.rec_method
    opt.cw_max_iterations = opt.cw_rec_max_iter
    opt.cw_confidence = opt.cw_rec_conf
    opt.pgd_eps = opt.pgd_eps_rec
    opt.pgd_eps_iter = opt.pgd_eps_iter_rec
    opt.pgd_iterations = opt.pgd_iterations_rec
    attack_label = label

    perturbed_data = adversarialattack(opt.attack, classifier, data, attack_label, opt)
    opt.attack = attack
    opt.cw_max_iterations = cw_max_iterations
    opt.cw_confidence = cw_confidence
    opt.pgd_eps = pgd_eps
    opt.pgd_eps_iter = pgd_eps_iter
    opt.pgd_iterations = pgd_iterations
    return perturbed_data


@torch.enable_grad()
def getAdv(classifier, data, label):
    if opt.attack == 'Data':
        return data
    data = data.detach().clone()
    data.requires_grad = True
    clean_pred = classifier(data).max(1)[1]
    targeted = opt.attack.endswith('T')
    if targeted:
        targeted_label = []
        # Randomly choose targeted label that is not the ground truth one
        for i in range(data.size(0)):
            targeted_label.append(randint(1, opt.classifier_classes - 1))
        attack_label = torch.fmod(label + torch.tensor(targeted_label).long().to(opt.device), opt.classifier_classes)
    else:
        # Untargeted attacks use the model classification labels
        attack_label = label

    # perturbed_data = white_cw_attack(classifier, data, attack_label,opt.classifier_classes,
    #                 targeted=targeted, learning_rate=opt.cw_lr, max_iterations=opt.cw_max_iterations,
    #                 confidence=opt.cw_confidence,attack_type=opt.loss_type)
    perturbed_data = adversarialattack(opt.attack, classifier, data, attack_label, opt)
    return perturbed_data


def get_aug(data, eps):
    return data + torch.randn(data.size()).cuda(device=opt.device) * eps


from torch.utils.data import Dataset, DataLoader, TensorDataset


def generate_interpret_only(opt):
    if opt.interpret_method == 'org':
        return
    writer = SummaryWriter(opt.summary_name)
    # Initialize the network
    # choose classifier_net in package:models
    # classifier_net = 'vgg11bn'
    if 'imagenet' in opt.dataset:
        classifier_net = 'wide_resnet'
    elif 'arcface' in opt.dataset:
        classifier_net = 'arcfacebn'
    else:
        classifier_net = 'vgg11bn'
    if 'black' in opt.attack_box:
        classifier_net = opt.classifier_net
    saved_name = f'../classifier_pth/classifier_{classifier_net}_{opt.dataset}_best.pth'
    model = eval(classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,
                                 dataset=opt.dataset)
    print('using network {}'.format(classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name, map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')

    CleanInterpret = []
    AdvInterpret = []

    for d_name, i_name in [('AdvDataSet', 'AdvInterpret'), ('CleanDataSet', 'CleanInterpret')]:
        dataset = eval(d_name)
        dataset = TensorDataset(dataset, Labels)
        dataloader = DataLoader(dataset=dataset, batch_size=opt.val_batchsize, shuffle=False)

        correct = 0
        global_step = 0
        size = len(dataloader.dataset)
        # Loop over all examples in test set
        for data, target in tqdm(dataloader, desc=d_name):
            # Send the data and label to the device
            data, target = data.to(opt.device), target.to(opt.device)
            batch_size = data.shape[0]

            # Forward pass the data through the model
            output = model(data)
            output = predict_batch(output, opt)
            # init_pred = output.max(1)[1]
            init_pred = output
            correct += torch.sum(init_pred == target.data)
            writer.add_scalar(d_name + 'correct', torch.sum(init_pred == target.data).item() / batch_size,
                              global_step=global_step)
            global_step += 1

            interpreter = interpretermethod(model, opt.interpret_method)
            saliency_images = interpreter.interpret(data)
            interpreter.release()
            eval(i_name).append(saliency_images.clone().cpu())
        print("{}_correct = {:.4f}".format(d_name, correct.item() / size))

    try:
        os.makedirs(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}')
    except Exception as e:
        print(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}  already exist')

    try:
        os.makedirs(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}')
    except Exception as e:
        print(
            f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}  already exist')

    for v in ['CleanInterpret', 'AdvInterpret']:
        n = eval(v)
        n = torch.cat(n, dim=0)
        torch.save(n,
                   f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/{v}.npy')

    return


def generate_result_only(opt):
    writer = SummaryWriter(opt.summary_name)
    # Initialize the network
    # choose classifier_net in package:models
    # classifier_net = 'vgg11bn'
    if 'imagenet' in opt.dataset:
        classifier_net = 'wide_resnet'
    elif 'arcface' in opt.dataset:
        classifier_net = 'arcfacebn'
    else:
        classifier_net = 'vgg11bn'
    if 'black' in opt.attack_box:
        classifier_net = opt.classifier_net
    saved_name = f'../classifier_pth/classifier_{classifier_net}_{opt.dataset}_best.pth'
    model = eval(classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,
                                 dataset=opt.dataset)
    print('using network {}'.format(classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name, map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')

    CleanResult = []
    AdvResult = []

    for d_name, i_name in [('AdvDataSet', 'AdvResult'), ('CleanDataSet', 'CleanResult')]:
        dataset = eval(d_name)
        dataset = TensorDataset(dataset, Labels)
        dataloader = DataLoader(dataset=dataset, batch_size=opt.val_batchsize, shuffle=False)

        correct = 0
        global_step = 0
        size = len(dataloader.dataset)
        # Loop over all examples in test set
        for data, target in tqdm(dataloader, desc=d_name):
            # Send the data and label to the device
            data, target = data.to(opt.device), target.to(opt.device)
            batch_size = data.shape[0]

            # Forward pass the data through the model
            output = model(data)
            output = predict_batch(output, opt)
            # init_pred = output.max(1)[1]  # get the index of the max log-probability
            init_pred = output
            correct += torch.sum(init_pred == target.data)
            writer.add_scalar(d_name + 'correct', torch.sum(init_pred == target.data).item() / batch_size,
                              global_step=global_step)
            global_step += 1
            eval(i_name).append(init_pred.clone().cpu())
        print("{}_correct = {:.4f}".format(d_name, correct.item() / size))

    try:
        os.makedirs(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}')
    except Exception as e:
        print(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}  already exist')

    for v in ['CleanResult', 'AdvResult']:
        n = eval(v)
        n = torch.cat(n, dim=0)
        torch.save(n, f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')
    return


def generate():
    writer = SummaryWriter(opt.summary_name)
    if opt.data_phase == 'test':
        loader = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=False, workers=opt.workers,
                              shuffle=False)
    else:
        loader = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=True, workers=opt.workers,
                              shuffle=False)

    if opt.data_phase == 'test':
        adv_loader = build_loader(opt.data_root, "lfw-sticker", opt.train_batchsize, train=False, workers=opt.workers,
                                  shuffle=False)
    else:
        adv_loader = build_loader(opt.data_root, "lfw-sticker", opt.train_batchsize, train=True, workers=opt.workers,
                                  shuffle=False)
    classifier_net = 'arcfacebn'
    saved_name = f'../classifier_pth/classifier_{classifier_net}_{opt.dataset}_best.pth'
    model = eval(classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,
                                 dataset=opt.dataset)
    print('using network {}'.format(classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name, map_location=opt.device))
    model = model.to(opt.device)

    ## attack
    correct = 0
    org_correct = 0
    global_step = 0
    size = len(loader.dataset)
    print('dataset size is ', size)
    model.eval()

    CleanDataSet = []
    AdvDataSet = []
    Labels = []

    num = 0
    # Loop over all examples in test set
    for data, target in tqdm(loader, desc=opt.data_phase):
        # Send the data and label to the device
        data, target = data.to(opt.device), target.to(opt.device)
        batch_size = data.shape[0]
        output = model(data)
        output = predict_batch(output, opt)
        # init_pred = output.max(1)[1]
        init_pred = output
        org_correct += torch.sum(init_pred == target.data)
        # writer.add_scalar('org_correct',torch.sum(init_pred == target.data).item()/batch_size,global_step=global_step)

        global_step += 1

        # Get data
        CleanDataSet.append(data.clone().cpu())
        Labels.append(target.clone().cpu())

        num += batch_size
        if org_correct.item() > opt.max_num:
            break

    print(num, org_correct / num)

    num = 0
    adv_global_step = 0
    for adv_data, target in tqdm(adv_loader, desc=opt.data_phase):
        # Send the data and label to the device
        adv_data, target = adv_data.to(opt.device), target.to(opt.device)
        batch_size = adv_data.shape[0]
        # Forward pass the data through the model
        output = model(adv_data)
        output = predict_batch(output, opt)
        # init_pred = output.max(1)[1]  # get the index of the max log-probability
        init_pred = output
        correct += torch.sum(init_pred == target.data)
        # writer.add_scalar('adv_correct',torch.sum(init_pred == target.data).item()/batch_size,global_step=global_step)

        adv_global_step += 1
        AdvDataSet.append(adv_data.clone().cpu())

        num += batch_size
        if adv_global_step >= global_step:
            break

    print(num, correct / num)

    try:
        os.makedirs(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}')
    except Exception as e:
        print(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}  already exist')

    for v in ['CleanDataSet', 'AdvDataSet', 'Labels']:
        n = eval(v)
        n = torch.cat(n, dim=0)
        torch.save(n, f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')

    return


def generate_black():
    writer = SummaryWriter(opt.summary_name)
    if opt.data_phase == 'test':
        loader = pairs_loader(f'../data/{opt.dataset}/{opt.data_phase}_pair.pth', opt.train_batchsize,
                              workers=opt.workers, shuffle=True)
    else:
        loader = pairs_loader(f'../data/{opt.dataset}/{opt.data_phase}_pair.pth', opt.train_batchsize,
                              workers=opt.workers, shuffle=False)

    # Initialize the network
    # choose classifier_net in package:models
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels,
                                     num_classes=opt.classifier_classes, dataset=opt.dataset)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name, map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    attacker = blackboxAttacker(opt.attack, model, opt)

    ## attack
    correct = 0
    org_correct = 0
    global_step = 0
    size = len(loader.dataset)
    print('dataset size is ', size)

    CleanDataSet = []
    AdvDataSet = []
    Labels = []

    num = 0
    _pnt = True
    # Loop over all examples in test set
    for data, target, pair, ptarget in tqdm(loader, desc=opt.data_phase + "_" + opt.attack):
        # Send the data and label to the device
        data, target = data.to(opt.device), target.to(opt.device)
        pair, ptarget = pair.to(opt.device), ptarget.to(opt.device)
        batch_size = data.shape[0]
        if _pnt:
            print(batch_size)
            _pnt = False
        adv_data = attacker(data, target, pair, ptarget)

        # Forward pass the data through the model
        output = model(adv_data)
        init_pred = output.max(1)[1]  # get the index of the max log-probability
        correct += torch.sum(init_pred == target.data)
        writer.add_scalar('adv_correct', torch.sum(init_pred == target.data).item() / batch_size,
                          global_step=global_step)

        output = model(data)
        init_pred = output.max(1)[1]
        org_correct += torch.sum(init_pred == target.data)
        writer.add_scalar('org_correct', torch.sum(init_pred == target.data).item() / batch_size,
                          global_step=global_step)

        global_step += 1

        # Get data
        CleanDataSet.append(data.clone().cpu())
        Labels.append(target.clone().cpu())
        AdvDataSet.append(adv_data.clone().cpu())

        num += batch_size
        if num > opt.max_num:
            break

    print("num = {} , adv_correct = {:.4f} , org_correct = {:.4f}".format(num, correct.item() / num,
                                                                          org_correct.item() / num))

    try:
        os.makedirs(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}')
    except Exception as e:
        print(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}  already exist')

    for v in ['CleanDataSet', 'AdvDataSet', 'Labels']:
        n = eval(v)
        n = torch.cat(n, dim=0)
        torch.save(n, f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')

    return


def save_roc_cruve(y, prob, path):
    try:
        os.makedirs(path)
    except:
        pass
    pk_dump(y, os.path.join(path, 'y'))
    pk_dump(prob, os.path.join(path, 'prob'))


import sklearn


def save_roc_fig(y, prob, path, title='Receiver operating characteristic example'):
    fpr, tpr, thresholds = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = sklearn.metrics.auc(fpr, tpr)  ###计算auc的值
    threshold, point = Find_Optimal_Cutoff(tpr, fpr, thresholds)

    lw = 2
    # plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    plt.plot(point[0], point[1], 'ro')
    plt.savefig(path)
    plt.close()


def redraw():
    Z = pk_load(
        f'../roc_cruve_data/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/PCA-{opt.detect_method}/Normal/prob')
    Y = pk_load(
        f'../roc_cruve_data/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/PCA-{opt.detect_method}/Normal/y')
    save_roc_fig(Y, Z,
                 f'../roc_cruve_data/figs/pca/{opt.data_phase}_{opt.attack}_{opt.attack_param}_{opt.interpret_method}_PCA-{opt.detect_method}_Normal.png',
                 title=f'{opt.data_phase}_{opt.attack}_{opt.attack_param}_{opt.interpret_method}_PCA-{opt.detect_method}_AUC曲线')

    Z = pk_load(
        f'../roc_cruve_data/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/PCA-{opt.detect_method}/Reverse/prob')
    Y = pk_load(
        f'../roc_cruve_data/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/PCA-{opt.detect_method}/Reverse/y')
    save_roc_fig(Y, Z,
                 f'../roc_cruve_data/figs/pca/{opt.data_phase}_{opt.attack}_{opt.attack_param}_{opt.interpret_method}_PCA-{opt.detect_method}_Reverse.png',
                 title=f'{opt.data_phase}_{opt.attack}_{opt.attack_param}_{opt.interpret_method}_PCA-{opt.detect_method}_AUC曲线')


def get_sift_data(opt):
    print("generate_result_only")
    generate_result_only(opt)
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    if opt.interpret_method == 'org':
        CleanInterpret = CleanDataSet
        AdvInterpret = AdvDataSet
    else:
        try:
            CleanInterpret = torch.load(
                f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
            AdvInterpret = torch.load(
                f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')
        except:
            generate_interpret_only(opt)
            CleanInterpret = torch.load(
                f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
            AdvInterpret = torch.load(
                f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')

    CleanResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanResult.npy')
    AdvResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvResult.npy')

    CleanDataSet = CleanDataSet[CleanResult == Labels]
    AdvDataSet = AdvDataSet[(CleanResult == Labels) & (AdvResult != Labels)]
    CleanInterpret = CleanInterpret[CleanResult == Labels]
    AdvInterpret = AdvInterpret[(CleanResult == Labels) & (AdvResult != Labels)]
    CleanLabels = Labels[CleanResult == Labels]
    AdvLabels = Labels[(CleanResult == Labels) & (AdvResult != Labels)]
    if opt.rec_max is not None and opt.rec_max > 0:
        rec_max = min([opt.rec_max, CleanDataSet.shape[0], AdvDataSet.shape[0]])
        CleanDataSet = CleanDataSet[:rec_max]
        AdvDataSet = AdvDataSet[:rec_max]
        CleanInterpret = CleanInterpret[:rec_max]
        AdvInterpret = AdvInterpret[:rec_max]
        CleanLabels = CleanLabels[:rec_max]
        AdvLabels = AdvLabels[:rec_max]
    return CleanDataSet, AdvDataSet, CleanInterpret, AdvInterpret, Labels, CleanLabels, AdvLabels


def get_difference(p):
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    x = CleanDataSet - AdvDataSet
    x = x.view(x.shape[0], -1)
    l = torch.norm(x, dim=1, p=p).mean().item() / x.shape[1]
    return l


def auc_curve(y, prob, plt=True):
    fpr, tpr, thresholds = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = sklearn.metrics.auc(fpr, tpr)  ###计算auc的值
    threshold, point = Find_Optimal_Cutoff(tpr, fpr, thresholds)

    if plt:
        lw = 2
        # plt.figure(figsize=(10,10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        plt.plot(point[0], point[1], 'ro')

        plt.show()

    return roc_auc, threshold, thresholds


# 约登数寻找阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def make_graph():
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    CleanInterpret = torch.load(
        f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
    AdvInterpret = torch.load(
        f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')
    size = CleanInterpret.shape[0]

    for v in ['CleanDataSet', 'AdvDataSet', 'CleanInterpret', 'AdvInterpret']:
        n = eval(v)
        try:
            os.makedirs(f'../maps/{opt.interpret_method}/{opt.attack}_{opt.attack_param}')
        except Exception as e:
            print(f'../maps/{opt.interpret_method}/{opt.attack}_{opt.attack_param}  already exist')

        vutils.save_image(n[:64], f'../maps/{opt.interpret_method}/{opt.attack}_{opt.attack_param}/{v}.jpg', nrow=8)

    x = CleanDataSet - AdvDataSet
    x = x.view(size, -1)
    l2 = torch.norm(x, dim=1, p=2).mean().item()
    print('l2 distance is {:.4f}'.format(l2))


def make_dir(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(f'{path}  already exist')


def plot_auc_eps(data, path, methods=['LRP', 'org', 'VG', 'GBP', 'IG', 'attack_rate']):
    for method in methods:
        x = []
        y = []
        for eps, item in data.items():
            x.append(eps)
            y.append(item[method])
        plt.plot(x, y, label=method)
    plt.xlabel('eps')
    plt.ylabel('auc')
    plt.title(path)
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.close()


def get_difference(p):
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    x = CleanDataSet - AdvDataSet
    x = x.view(x.shape[0], -1)
    l = torch.norm(x, dim=1, p=p).mean().item()
    return l


def generate_all():
    print("generate_all")
    if 'FGSM' in opt.attack:
        opt.fgsm_eps = opt.attack_param
        generate()
    elif 'PGD' in opt.attack:
        opt.pgd_eps = opt.attack_param
        generate()
    elif 'CW' in opt.attack:
        opt.cw_confidence = opt.attack_param
        generate()
    elif 'DDN' in opt.attack:
        opt.dnn_steps = opt.attack_param
        generate()
    elif 'DFool' in opt.attack:
        opt.dfool_overshoot = opt.attack_param
        generate()
    elif 'ROA' in opt.attack:
        generate()
    elif "Boundary" in opt.attack:
        generate_black()
    elif "NES" in opt.attack:
        generate_black()
    elif "Optim" in opt.attack:
        generate_black()

    generate_result_only(opt)
    for opt.interpret_method in opt.interpret_methods:
        generate_interpret_only(opt)
    CleanDataSet, AdvDataSet, CleanInterpret, AdvInterpret, _, _, _ = get_sift_data(opt)
    attack_rate = AdvInterpret.shape[0] / CleanInterpret.shape[0]
    l2 = get_difference(2)
    print('attack is {}, param is {:.4f}, attack rate is {:.4f}, l2 is {:.4f}'.format(
        opt.attack, opt.attack_param, attack_rate, l2
    ))


def get_reducer(interpret_methods, opt):
    rec_max = opt.rec_max
    opt.rec_max = None
    reducers = {}
    for opt.interpret_method in interpret_methods:
        if 'black' in opt.attack_box and 'imagenet' in opt.dataset:
            reducer = eval(opt.reducer)(n_components=None)
        else:
            reducer = eval(opt.reducer)()
        CleanDataSet, AdvDataSet, CleanInterpret, AdvInterpret, _, _, _ = get_sift_data(opt)
        reducer.train(CleanInterpret)
        reducers[opt.interpret_method] = reducer
    opt.rec_max = rec_max
    return reducers


from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import sklearn


def detect(opt, interpret_methods, reverse_list=[], train=True):
    ensembled_clf = Ensembler(interpret_methods=interpret_methods, detect_method=opt.detect_method, use_voting=False)
    clf_path_name = f'../classifier_pth/{opt.classifier_net}_{opt.dataset}_{opt.detect_method}_clf-ensemble'
    reducers = get_reducer(interpret_methods, opt=opt)

    reverse = {}
    for i in interpret_methods:
        reverse[i] = False
    for i in reverse_list:
        reverse[i] = True

    for opt.interpret_method in interpret_methods:
        reducer = reducers[opt.interpret_method]
        CleanDataSet, AdvDataSet, CleanInterpret, AdvInterpret, _, _, _ = get_sift_data(opt)
        cleansize = CleanInterpret.shape[0]
        advsize = AdvInterpret.shape[0]
        total = cleansize + advsize
        print(
            f"i_method is {opt.interpret_method}, total is {total}, clean size is {cleansize} (clean/total={cleansize}/{total}), adv size is {advsize} (adv/clean={advsize}/{cleansize})")

        if train:
            print('training begin')
            t1 = datetime.now()
            X_train = reducer.reduce(CleanInterpret)
            reducer.dump(os.path.join(clf_path_name, f"reducer_{opt.data_phase}_{opt.interpret_method}.npy"))
            ensembled_clf.fit(opt.interpret_method, X_train)
            t2 = datetime.now()
            totalseconds = (t2 - t1).total_seconds()
            h = totalseconds // 3600
            m = (totalseconds - h * 3600) // 60
            s = totalseconds - h * 3600 - m * 60
            print('training end in time {}h {}m {:.4f}s'.format(h, m, s))
            ensembled_clf.save(clf_path_name)
        else:
            # if not os.path.exists(os.path.join(clf_path_name,f"reducer_{opt.data_phase}_{opt.interpret_method}.npy")):
            # print(f"train reducer {opt.data_phase}")
            # X_train = reducer.train(CleanInterpret)
            # reducer.dump(os.path.join(clf_path_name,f"reducer_{opt.data_phase}_{opt.interpret_method}.npy"))
            # reducer.load(os.path.join(clf_path_name,f"reducer_{opt.data_phase}_{opt.interpret_method}.npy"))
            # X_train = reducer.reduce(CleanInterpret)
            X_train = reducer.reduce(CleanInterpret)
            ensembled_clf.load(clf_path_name)

        print('eval begin')
        t1 = datetime.now()
        InterpretAll = torch.cat([CleanInterpret, AdvInterpret], dim=0)
        org_size = CleanInterpret.shape[0]
        adv_size = AdvInterpret.shape[0]
        InterpretAll = reducer.reduce(InterpretAll)
        if 'cifar' in opt.dataset:
            norm_method = 'min-max'
        else:
            norm_method = 'mean-std'

        ensembled_clf.calculate(opt.interpret_method, InterpretAll, norm_method=norm_method,
                                reverse=reverse[opt.interpret_method])
        t2 = datetime.now()
        totalseconds = (t2 - t1).total_seconds()
        h = totalseconds // 3600
        m = (totalseconds - h * 3600) // 60
        s = totalseconds - h * 3600 - m * 60
        print('eval end in time {}h {}m {:.4f}s'.format(h, m, s))

    Z = ensembled_clf.ensemble(method="min")
    Z_n = Z.copy()
    Y = torch.cat([torch.ones(cleansize), torch.zeros(advsize)], dim=0).numpy()
    roc_auc, threshold, thresholds = auc_curve(Y, Z, plt=False)
    th_n = threshold
    acc = (Z[:cleansize] >= threshold).sum() + (Z[cleansize:] < threshold).sum()
    choose_n = (Z_n[:] < threshold)
    acc = acc / (cleansize + advsize)
    print('{}_{} :: X-Det :: Normal :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(opt.dataset,
                                                                                                      opt.classifier_net,
                                                                                                      roc_auc,
                                                                                                      threshold, acc))
    roc_auc_nrom = roc_auc

    print('normal choose acc = {:.4f}'.format((cleansize + advsize - (choose_n == Y).sum()) / (cleansize + advsize)))

    Z = ensembled_clf.ensemble(method='max')
    Z_r = Z.copy()
    Y = torch.cat([torch.zeros(cleansize), torch.ones(advsize)], dim=0).numpy()
    roc_auc, threshold, thresholds = auc_curve(Y, Z, plt=False)
    th_r = threshold
    acc = (Z[:cleansize] < threshold).sum() + (Z[cleansize:] >= threshold).sum()
    choose_r = (Z_r[:] >= threshold)
    acc = acc / (cleansize + advsize)
    print('{}_{} :: X-Det :: Reverse :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(opt.dataset,
                                                                                                       opt.classifier_net,
                                                                                                       roc_auc,
                                                                                                       threshold, acc))
    roc_auc_rev = roc_auc

    print('reverse choose acc = {:.4f}'.format((choose_r == Y).sum() / (cleansize + advsize)))

    if roc_auc_nrom > roc_auc_rev:
        auc_min = roc_auc_nrom
        return auc_min, choose_n, ensembled_clf, 'min', th_n, reducers
    else:
        auc_max = roc_auc_rev
        return auc_max, choose_r, ensembled_clf, 'max', th_r, reducers


def test_to_rem(dataset, label, model):
    dataset = TensorDataset(dataset, label)
    loader = DataLoader(dataset=dataset, batch_size=opt.val_batchsize, shuffle=False)
    model.eval()

    running_corrects = 0
    # Iterate over data
    size = len(loader.dataset)

    for inputs, labels in tqdm(loader, desc=f'remainted_{opt.dataset}_{opt.classifier_net}'):
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)

        outputs = model(inputs)
        preds = outputs.max(1)[1]

        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects / size

    print('train Acc: {:.4f}'.format(epoch_acc.item()))

    return epoch_acc.item(), running_corrects.item(), size


def generate_interpret_only_rec(lb):
    if opt.interpret_method == 'org':
        return
    # Initialize the network
    # choose classifier_net in package:models
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels,
                                     num_classes=opt.classifier_classes, dataset=opt.dataset)
    print('generate_interpret_only_rec using network {}'.format(opt.classifier_net))
    print('generate_interpret_only_rec loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name, map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    dataset = torch.load(
        f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Rec-{lb}.npy')
    interpret = []

    dataset = TensorDataset(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.val_batchsize, shuffle=False)

    for data, in tqdm(dataloader, desc=f'generate_interpret_only_rec-{lb}'):
        # Send the data and label to the device
        data = data.to(opt.device)
        batch_size = data.shape[0]

        # Forward pass the data through the model
        output = model(data)
        interpreter = interpretermethod(model, opt.interpret_method)
        saliency_images = interpreter.interpret(data)
        interpreter.release()
        interpret.append(saliency_images.clone().cpu())

    make_dir(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}')

    n = torch.cat(interpret, dim=0)
    torch.save(n,
               f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/wholeLine-Rec-{lb}.npy')
    return


def get_Z(ensembled_clf, reducers, interpret_methods=['DL', 'LRP', 'VG', 'GBP', 'IG', 'org']):
    for lb in range(opt.classifier_classes):
        for opt.interpret_method in interpret_methods:
            if opt.interpret_method == 'org':
                interpret = torch.load(
                    f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Rec-{lb}.npy')
            else:
                interpret = torch.load(
                    f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/wholeLine-Rec-{lb}.npy')
            reducer = reducers[opt.interpret_method]
            clf = ensembled_clf.find_clf(opt.interpret_method)
            size = interpret.shape[0]
            X = reducer.reduce(interpret)
            Z = clf.decision_function(X)
            make_dir(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}')
            np.save(
                f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/wholeLine-Rec-Z-{lb}.npy',
                Z)


def generate_rectified(model, ds, labels, interpret_methods=['DL', 'LRP', 'VG', 'GBP', 'IG', 'org']):
    for lb in range(opt.classifier_classes):
        dataset = TensorDataset(ds, labels)
        rec = []
        rec_result = []
        dataloader = DataLoader(dataset=dataset, batch_size=opt.val_batchsize, shuffle=False)
        for d, l in tqdm(dataloader, desc=f'wholeLine - rec - {lb}'):
            d = d.to(opt.device)
            tmp_label = l.clone()
            tmp_label[:] = lb
            tmp_label = tmp_label.to(opt.device)
            rec_img = getRec(model, d, tmp_label)
            rr = model(rec_img)
            rr = rr.max(1)[1]
            rec.append(rec_img.clone().detach().cpu())
            rec_result.append(rr)
        rec = torch.cat(rec).cpu()
        rec_result = torch.cat(rec_result).cpu()
        print('{} rec shape is {}'.format('wholeLine', rec.shape))
        torch.save(rec, f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Rec-{lb}.npy')
        torch.save(rec_result,
                   f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Rec_result-{lb}.npy')

    for lb in range(opt.classifier_classes):
        for opt.interpret_method in interpret_methods:
            if opt.interpret_method == 'org':
                continue
            generate_interpret_only_rec(lb)


def get_normers(interpret_methods=['DL', 'LRP', 'VG', 'GBP', 'IG', 'org']):
    normers = {}
    for opt.interpret_method in interpret_methods:
        normalizer = Normalizer()
        normalizer.load(
            f'../classifier_pth/{opt.classifier_net}_{opt.dataset}_{opt.detect_method}_clf-ensemble/normer/{opt.interpret_method}')
        normers[opt.interpret_method] = normalizer
    return normers


def draw_rec_z(interpret_methods=['DL', 'LRP', 'VG', 'GBP', 'IG', 'org'], reverse_list=[]):
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels,
                                     num_classes=opt.classifier_classes, dataset=opt.dataset)
    print('draw_rec_z using network {}'.format(opt.classifier_net))
    print('draw_rec_z loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name, map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    dataset = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Dataset.npy')
    label = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Label.npy')
    normers = get_normers(interpret_methods)
    reverse = {}
    for i in interpret_methods:
        reverse[i] = False
    for i in reverse_list:
        reverse[i] = True
    Rec = {}
    Z_adv = {}
    Rec_res = {}
    for lb in range(opt.classifier_classes):
        print('loading data', lb)
        Rec[lb] = torch.load(
            f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Rec-{lb}.npy')
        Rec_res[lb] = torch.load(
            f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Rec_result-{lb}.npy').cpu()
        for opt.interpret_method in interpret_methods:
            Z_tmp = np.load(
                f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/wholeLine-Rec-Z-{lb}.npy')
            # Z_adv[(lb,opt.interpret_method)] = normers[opt.interpret_method].get(Z_tmp,method = 'mean-std',reverse=reverse[opt.interpret_method])
            Z_adv[(lb, opt.interpret_method)] = Z_tmp

    def test_inner(dataset, labels, phase):
        dataset = TensorDataset(dataset, labels)
        dataloader = DataLoader(dataset=dataset, batch_size=opt.val_batchsize, shuffle=False)
        correct = 0
        global_step = 0
        size = len(dataloader.dataset)
        # Loop over all examples in test set
        c = []
        r = []
        for data, target in tqdm(dataloader, desc=phase):
            # Send the data and label to the device
            data, target = data.to(opt.device), target.to(opt.device)
            batch_size = data.shape[0]

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1)[1]  # get the index of the max log-probability
            correct += torch.sum(init_pred == target.data)
            c.append(init_pred == target.data)
            r.append(init_pred)
        print("{}_correct = {:.4f}".format(phase, correct.item() / size))
        c = torch.cat(c).cpu()
        r = torch.cat(r).cpu()
        return c, r

    _, AdvResult = test_inner(dataset, label, 'test_adv_result')
    size = Rec_res[0].shape[0]
    not_rectified = torch.zeros(Rec[0].shape[0], dtype=torch.bool)
    target_lbs = torch.zeros(Rec[0].shape[0])
    Z_rec = {}
    Z_rec_en = []
    Z_true = []
    for lb in range(opt.classifier_classes):
        Z_rec[lb] = []

    for i in range(size):
        Zs = []
        Zind = []
        last = None
        for lb in range(opt.classifier_classes):
            Z = []
            for opt.interpret_method in interpret_methods:
                Z.append(Z_adv[(lb, opt.interpret_method)][i])
            last = max(Z)
            if (label[i].item() == lb):
                Z_true.append(last)
            Z_rec[lb].append(last)
            if AdvResult[i] == Rec_res[lb][i]:
                continue
            Zs.append(last)
            Zind.append(lb)
        if len(Zs) == 0:
            Zs = [last]
            Zind = [randint(0, opt.classifier_classes - 1)]
            not_rectified[i] = True
        m = min(Zs)
        Z_rec_en.append(m)
        ind = Zs.index(m)
        target_lbs[i] = Zind[ind]

    plt.figure(figsize=(25, 10))
    y = np.arange(size)
    for lb in range(opt.classifier_classes):
        size = Rec_res[lb].shape[0]
        x = Z_rec[lb]
        plt.scatter(y, x)
    plt.plot(y, Z_rec_en, 'g')
    plt.plot(y, Z_true, 'r')
    plt.savefig(f'../{opt.tmp_dataset}/attack_eps/{opt.data_phase}_{opt.attack}_{opt.attack_param}_wholeLine-Zmap.png')
    plt.close()


def test_to_rec(reducers, ensembled_clf, clf_method, th, dataset, label, model,
                interpret_methods=['DL', 'LRP', 'VG', 'GBP', 'IG', 'org'], reverse_list=[]):
    generate_rectified(model, dataset, label, interpret_methods)
    get_Z(ensembled_clf, reducers, interpret_methods)
    torch.save(dataset, f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Dataset.npy')
    torch.save(label, f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Label.npy')

    dataset = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Dataset.npy')
    label = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Label.npy')

    normers = get_normers(interpret_methods)
    reverse = {}
    for i in interpret_methods:
        reverse[i] = False
    for i in reverse_list:
        reverse[i] = True

    Rec = {}
    Z_adv = {}
    Rec_res = {}
    for lb in range(opt.classifier_classes):
        print('loading data', lb)
        Rec[lb] = torch.load(
            f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Rec-{lb}.npy')
        Rec_res[lb] = torch.load(
            f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/wholeLine-Rec_result-{lb}.npy').cpu()
        for opt.interpret_method in interpret_methods:
            Z_tmp = np.load(
                f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/wholeLine-Rec-Z-{lb}.npy')
            Z_adv[(lb, opt.interpret_method)] = normers[opt.interpret_method].get(Z_tmp, method='mean-std',
                                                                                  reverse=reverse[opt.interpret_method])

    def test_inner(dataset, labels, phase):
        dataset = TensorDataset(dataset, labels)
        dataloader = DataLoader(dataset=dataset, batch_size=opt.val_batchsize, shuffle=False)
        correct = 0
        global_step = 0
        size = len(dataloader.dataset)
        # Loop over all examples in test set
        c = []
        r = []
        for data, target in tqdm(dataloader, desc=phase):
            # Send the data and label to the device
            data, target = data.to(opt.device), target.to(opt.device)
            batch_size = data.shape[0]

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1)[1]  # get the index of the max log-probability
            correct += torch.sum(init_pred == target.data)
            c.append(init_pred == target.data)
            r.append(init_pred)
        print("{}_correct = {:.4f}".format(phase, correct.item() / size))
        c = torch.cat(c).cpu()
        r = torch.cat(r).cpu()
        return c, r

    datarec = []
    not_rectified = torch.zeros(Rec[0].shape[0], dtype=torch.bool)
    target_lbs = torch.zeros(Rec[0].shape[0])

    _, AdvResult = test_inner(dataset, label, 'test_adv_result')
    size = AdvResult.shape[0]
    print(size)
    print(Rec[0].shape[0])

    best = not_rectified = torch.zeros(size, dtype=torch.bool)
    for lb in range(opt.classifier_classes):
        best[Rec_res[lb] == label] = True
    print('best condition is {:.4f}'.format(torch.sum(best).item() / size))

    for i in range(size):
        Zs = []
        Zind = []
        last = None
        if clf_method == 'min':
            for lb in range(opt.classifier_classes):
                Z = []
                for opt.interpret_method in interpret_methods:
                    Z.append(Z_adv[(lb, opt.interpret_method)][i])
                last = min(Z)
                if AdvResult[i] == Rec_res[lb][i]:
                    continue
                Zs.append(min(Z))
                Zind.append(lb)
            if len(Zs) == 0:
                Zs = [last]
                Zind = [randint(0, opt.classifier_classes - 1)]
                not_rectified[i] = True
            m = max(Zs)
            if m < th:
                not_rectified[i] = True
            ind = Zs.index(m)
            datarec.append(Rec[Zind[ind]][i:i + 1])
            target_lbs[i] = Zind[ind]
        else:
            for lb in range(opt.classifier_classes):
                Z = []
                for opt.interpret_method in interpret_methods:
                    Z.append(Z_adv[(lb, opt.interpret_method)][i])
                last = max(Z)
                if AdvResult[i] == Rec_res[lb][i]:
                    continue
                Zs.append(max(Z))
                Zind.append(lb)
            if len(Zs) == 0:
                Zs = [last]
                Zind = [randint(0, opt.classifier_classes - 1)]
                not_rectified[i] = True
            m = min(Zs)
            if m > th:
                not_rectified[i] = True
            ind = Zs.index(m)
            datarec.append(Rec[Zind[ind]][i:i + 1])
            target_lbs[i] = Zind[ind]
    datarec = torch.cat(datarec)
    c, r = test_inner(datarec, label, 'AdvRec')
    first = c.clone()
    second = c.clone()
    first_size = torch.sum(c).cpu().numpy()
    print('first size = ', torch.sum(c).cpu().numpy())
    not_rectified[c] = False
    print('drop1 size = ', torch.sum(not_rectified).cpu().numpy())
    not_rectified[target_lbs != r] = True
    # print('drop2 size = ',torch.sum(not_rectified).cpu().numpy())
    c[not_rectified] = True
    second[not_rectified] = False
    drop = torch.sum(not_rectified).cpu().numpy()

    ans = {
        'detected_total_count': size,
        'rectified_count': size - drop,
        'true_count': torch.sum(first).item(),
        'rec_true': torch.sum(second).item(),
        'nr_true': torch.sum(first).item() - torch.sum(second).item()
    }
    ans['rec_true/rectified_count'] = ans['rec_true'] / ans['rectified_count'] if ans['rectified_count'] != 0 else 0
    ans['nr_true/(detected_total_count-rectified_count)'] = ans['nr_true'] / (size - ans['rectified_count']) if (size -
                                                                                                                 ans[
                                                                                                                     'rectified_count']) != 0 else 0
    ans['true_count/detected_total_count'] = ans['true_count'] / size if size != 0 else 0

    return ans


def test_detect(choose, model):
    CleanDataSet, AdvDataSet, CleanInterpret, AdvInterpret, _, CleanLabels, AdvLabels = get_sift_data(opt)
    DS = torch.cat([CleanDataSet, AdvDataSet])
    LBS = torch.cat([CleanLabels, AdvLabels])
    choose = torch.tensor(choose)
    to_rec_dataset = DS[choose]
    to_rec_label = LBS[choose]
    to_rem_dataset = DS[~choose]
    to_rem_label = LBS[~choose]

    dataset = TensorDataset(to_rec_dataset, to_rec_label)
    loader = DataLoader(dataset=dataset, batch_size=opt.val_batchsize, shuffle=False)
    model.eval()

    running_corrects = 0
    # Iterate over data
    size = len(loader.dataset)

    for inputs, labels in tqdm(loader, desc=f'remainted_{opt.dataset}_{opt.classifier_net}'):
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)

        outputs = model(inputs)
        preds = outputs.max(1)[1]

        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects / size

    print('choosed Acc: {:.4f}'.format(epoch_acc.item()))


def main(interpret_methods=['DL', 'LRP', 'VG', 'GBP', 'IG', 'org'], reverse_list=[], train=True):
    auc_min, choose, ensembled_clf, clf_method, th, reducers = detect(opt, interpret_methods, reverse_list, train)

    tmp = opt.classifier_net
    if "cifar10" in opt.dataset:
        opt.classifier_net = "vgg11bn"
    if 'imagenet' in opt.dataset:
        opt.classifier_net = 'wide_resnet'
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels,
                                     num_classes=opt.classifier_classes, dataset=opt.dataset)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name, map_location=opt.device))
    model = model.to(opt.device)
    model.eval()
    opt.classifier_net = tmp

    test_detect(choose, model)

    CleanDataSet, AdvDataSet, CleanInterpret, AdvInterpret, _, CleanLabels, AdvLabels = get_sift_data(opt)
    DS = torch.cat([CleanDataSet, AdvDataSet])
    LBS = torch.cat([CleanLabels, AdvLabels])
    choose = torch.tensor(choose)
    print("choose rate is {}/{}, shape is {}".format(torch.sum(choose).item(), choose.shape[0], choose.shape))
    to_rec_dataset = DS[choose]
    to_rec_label = LBS[choose]
    to_rem_dataset = DS[~choose]
    to_rem_label = LBS[~choose]
    choose_np = choose.numpy()
    choose_df = pd.DataFrame(choose_np)
    choose_df.to_csv(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/choose.csv',
                     float_format='%.3f')

    item = test_to_rec(reducers, ensembled_clf, clf_method, th, to_rec_dataset, to_rec_label, model, interpret_methods,
                       reverse_list)
    item['remainted_acc'], item['remained_true'], item['remained_size'] = test_to_rem(to_rem_dataset, to_rem_label,
                                                                                      model)
    item['rate'] = (item['rec_true'] + item['remained_true']) / (item['remained_size'] + item['rectified_count'])
    return item


if __name__ == '__main__':
    print("whole_line.py")
    opt = Options().parse_arguments()
    opt.gpu_id = 0
    opt.device = torch.device("cuda:{}".format(opt.gpu_id))

    opt.weight_decay = 5e-4
    opt.num_epoches = 100
    opt.lr = 0.01
    opt.lr_step = 10
    opt.lr_gamma = 0.8

    opt.pgd_eps = 0.08
    opt.pgd_eps_iter = 0.0005
    opt.pgd_iterations = 500
    opt.fgsm_eps = 0.015
    opt.summary_name = '../summary/iforest'
    opt.workers = 4

    opt.classifier_net = 'cwnet'
    opt.interpret_method = 'VG'
    opt.data_phase = 'test'
    opt.detect_method = 'iforest'  # ['iforest','SVDD','LOF','Envelope'] # Envelope 太难跑
    opt.minus = False
    opt.scale = 1
    opt.use_voting = False

    opt.classifier_net = 'wide_resnet_small'
    opt.tmp_dataset = f'tmp_dataset_{opt.classifier_net}_{opt.dataset}'
    opt.image_channels = 3
    opt.train_batchsize = opt.val_batchsize = 64
    opt.dfool_max_iterations = 600
    # opt.reducer = PCA_Reducer

    opt.cw_rec_conf = 0.5
    opt.cw_rec_max_iter = 6
    opt.rec_method = 'ROA'
    opt.pgd_eps_rec = 0.05
    opt.pgd_eps_iter_rec = 0.1
    opt.pgd_iterations_rec = 10

    report_args(opt)

    opt.optim_iter = 50
    opt.optim_alpha = 0.2  # 二分查找的初始倍率
    opt.optim_beta = 0.05  # 每次旋转的角度
    opt.filter_g = 0.8
    opt.stop_g = 0.5
    opt.max_num = 100
    opt.interpret_methods = ['VG', 'IG', 'GBP', "DL", "org"]
    opt.reverse_list = []
    opt.attack_box = 'white'

    for opt.classifier_net, opt.dataset, opt.image_channels, opt.attack_box, attack_list in [
        ('arcfacebn', 'arcface', 3, 'white', [("ROA", 1)]),
    ]:
        if "arcface" in opt.dataset:
            opt.rec_max = 5000
            opt.train_batchsize = opt.val_batchsize = 4
            opt.detect_method = "iforest"
            opt.reducer = "PCA_Reducer"

        opt.tmp_dataset = f'tmp_dataset_{opt.classifier_net}_{opt.dataset}'
        auc_values = {}
        auc_values_train = {}
        train = True
        for opt.attack, opt.attack_param in attack_list:
            # train dataset
            opt.data_phase = 'train'
            generate()
            generate_result_only(opt)
            for opt.interpret_method in opt.interpret_methods:
                generate_interpret_only(opt)

            item = main(train=train, interpret_methods=opt.interpret_methods, reverse_list=opt.reverse_list)
            print(item)
            auc_values_train[f"{opt.attack}_{opt.attack_param}"] = item
            draw_rec_z(interpret_methods=opt.interpret_methods, reverse_list=opt.reverse_list)
