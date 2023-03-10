
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor,KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from detectors.advJudge import AdvJudge
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from Options import Options,report_args
import os
from datasets import build_loader
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
from attack_methods import adversarialattack
import re,math
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from interpreter_methods import interpretermethod
from torch.utils.data import Dataset, DataLoader, TensorDataset

warnings.filterwarnings("ignore")
from utils import *

NORMALIZE_IMAGES = {
    'mnist': ((0.1307,), (0.3081,)),
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'svhn': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))}

# NOISE_MAG_LIST = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
NOISE_MAG_LIST = [0.01,0.002,0.001]


def generate_result_only():
    # Initialize the network
    # choose classifier_net in package:models
    if "cifar10" in opt.dataset:
        tmp = opt.classifier_net
        opt.classifier_net = "vgg11bn"
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,dataset=opt.dataset)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()
    if "cifar10" in opt.dataset:
        opt.classifier_net = tmp

    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
        
    CleanResult = []
    AdvResult = []
    
    for d_name,i_name in [('AdvDataSet','AdvResult'),('CleanDataSet','CleanResult')]:
        dataset = eval(d_name)
        dataset = TensorDataset(dataset,Labels)
        dataloader = DataLoader(dataset = dataset, batch_size=opt.val_batchsize,shuffle=False)

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
            init_pred = output.max(1)[1] # get the index of the max log-probability
            correct += torch.sum(init_pred == target.data)
            global_step+=1
            eval(i_name).append(init_pred.clone().cpu())
        print("{}_correct = {:.4f}".format(d_name,correct.item()/size))

    try:
        os.makedirs(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}')
    except Exception as e:
        print(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}  already exist')

    for v in ['CleanResult','AdvResult']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')
    return

def generate():
    if opt.interpret_method == 'org':
        return
    if opt.data_phase == 'test':
        loader = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=False, workers=opt.workers,shuffle=True)
    else:
        loader = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=True, workers=opt.workers,shuffle=True)
    if "cifar10" in opt.dataset:
        tmp = opt.classifier_net
        opt.classifier_net = "vgg11bn"
    # Initialize the network
    # choose classifier_net in package:models
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,dataset=opt.dataset)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    if "cifar10" in opt.dataset:
        opt.classifier_net = tmp
    ## attack
    correct = 0
    org_correct = 0
    global_step = 0
    size = len(loader.dataset)
    print('dataset size is ',size)
    model.eval()
    
    CleanDataSet = []
    AdvDataSet = []
    Labels = []
    CleanInterpret = []
    AdvInterpret = []
    class_cnt = {}
    for i in range(opt.classifier_classes):
        class_cnt[i] = 0

    num = 0
    # Loop over all examples in test set
    for data, target in tqdm(loader, desc=opt.data_phase):            
        # Send the data and label to the device
        data, target = data.to(opt.device), target.to(opt.device)
        batch_size = data.shape[0]
        if opt.attack == 'rand':
            adv_data = get_aug(data,eps=0.05)
        else:
            adv_data = getAdv(model,data,target)

        # Forward pass the data through the model
        output = model(adv_data)
        init_pred = output.max(1)[1] # get the index of the max log-probability
        correct += torch.sum(init_pred == target.data)

        output = model(data)
        init_pred = output.max(1)[1]
        org_correct += torch.sum(init_pred==target.data)

        global_step+=1

        # Get data
        CleanDataSet.append(data.clone().cpu())
        Labels.append(target.clone().cpu())
        AdvDataSet.append(adv_data.detach().clone().cpu())
        num+=batch_size
        if num>opt.max_num:
            break
    

    lbs = torch.cat(Labels,dim=0)
    for i in range(lbs.shape[0]):
        t = int(lbs[i])
        class_cnt[t]+=1
    print(class_cnt)
    for i,c in class_cnt.items():
        if c>num/2:
            raise Exception("class cnt not ballance")

        
    print("num = {} , adv_correct = {:.4f} , org_correct = {:.4f}".format(num,correct.item()/num,org_correct.item()/num))

    try:
        os.makedirs(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}')
    except Exception as e:
        print(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}  already exist')

    try:
        os.makedirs(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}')
    except Exception as e:
        print(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}  already exist')
        
    for v in ['CleanDataSet','AdvDataSet','Labels']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')



    dataset = TensorDataset(torch.cat(AdvDataSet),torch.cat(Labels))
    dataloader = DataLoader(dataset = dataset, batch_size=opt.val_batchsize,shuffle=False)

    correct = 0
    size = len(dataloader.dataset)
    # Loop over all examples in test set
    for data, target in tqdm(dataloader, desc='adv retest'):
        # Send the data and label to the device
        data, target = data.to(opt.device), target.to(opt.device)
        batch_size = data.shape[0]

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1)[1] # get the index of the max log-probability
        correct += torch.sum(init_pred == target.data)
    print("{}_correct = {:.4f}".format('adv',correct.item()/size))

    return

def get_sift_data():
    # generate_result_only()
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    if opt.interpret_method=='org':
        CleanInterpret = CleanDataSet
        AdvInterpret = AdvDataSet
    else:
        CleanInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
        AdvInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')
    CleanResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanResult.npy')
    AdvResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvResult.npy')

    CleanDataSet = CleanDataSet[CleanResult==Labels]
    AdvDataSet = AdvDataSet[(CleanResult==Labels)&(AdvResult!=Labels)]
    CleanLabels = CleanResult[CleanResult==Labels]
    AdvLabels = AdvResult[(CleanResult==Labels)&(AdvResult!=Labels)]
    return CleanDataSet,AdvDataSet,CleanLabels,AdvLabels,Labels


def main(isdiff):
    num_labels=opt.classifier_classes
    opt.data_phase='train'
    CleanDataSet,AdvDataSet,CleanLabels,AdvLabels,Labels = get_sift_data()

    # merge loader
    ds = torch.cat([CleanDataSet,AdvDataSet])
    lb = torch.cat([CleanLabels,AdvLabels]).long()
    size = lb.shape[0]

    loader_clean = TensorDataset(CleanDataSet,CleanLabels)
    loader_clean = DataLoader(dataset = loader_clean, batch_size=opt.val_batchsize,shuffle=True)

    # load model
    tmp = opt.classifier_net
    if "cifar10" in opt.dataset:
        opt.classifier_net = "vgg11bn"
    if 'imagenet' in opt.dataset:
        opt.classifier_net = 'wide_resnet'
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,dataset=opt.dataset)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()
    opt.classifier_net = tmp

    # fit
    advJudger = AdvJudge(model,opt.device,isdiff)
    advJudger.fit(loader_clean)

    # test
    opt.data_phase='test'
    CleanDataSet,AdvDataSet,CleanLabels,AdvLabels,Labels = get_sift_data()
    Z_org = []
    Z_adv = []
    for d_name,i_name in [('AdvDataSet','Z_adv'),('CleanDataSet','Z_org')]:
        dataset = eval(d_name)
        dataset = TensorDataset(dataset)
        dataloader = DataLoader(dataset = dataset, batch_size=opt.val_batchsize,shuffle=False)
        size = len(dataloader.dataset)
        # Loop over all examples in test set
        for data, in tqdm(dataloader, desc=d_name):
            # Send the data and label to the device
            data = data.to(opt.device)
            Z = advJudger.forward(data)
            Z = torch.tensor(Z)
            eval(i_name).append(Z)

    Z_org = torch.cat(Z_org)
    Z_adv = torch.cat(Z_adv)
    cleansize = Z_org.shape[0]
    advsize = Z_adv.shape[0]
    Z = torch.cat([Z_org,Z_adv],dim=0).numpy()
    Y = torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy()
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]>=threshold).sum() + (Z[cleansize:]<threshold).sum()
    acc = acc/(cleansize+advsize)
    print(f"advJudge_detect-D-{isdiff} test for attack = {opt.attack} param = {opt.attack_param}, auc score = {roc_auc}({1-roc_auc}), acc = {acc}")
    return roc_auc

def recover():
    opt.data_phase='test'
    CleanDataSet,AdvDataSet,_,_,Labels = get_sift_data()
    CleanResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanResult.npy')
    AdvResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvResult.npy')
    Lb = Labels[(CleanResult==Labels)&(AdvResult!=Labels)]
    
    loader = TensorDataset(AdvDataSet,Lb)
    loader = DataLoader(dataset = loader, batch_size=opt.val_batchsize,shuffle=True)

    # load model
    tmp = opt.classifier_net
    if "cifar10" in opt.dataset:
        opt.classifier_net = "vgg11bn"
    if 'imagenet' in opt.dataset:
        opt.classifier_net = 'wide_resnet'
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,dataset=opt.dataset)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()
    opt.classifier_net = tmp

    advJudger = AdvJudge(model,opt.device,False)

    return advJudger.recover(loader)
    

if __name__ == '__main__':
    print("advJudge_detect.py")
    opt = Options().parse_arguments()
    t1 = datetime.now()
    opt.dataset = 'cifar10'
    # opt.dataset = 'fmnist'
    opt.gpu_id = 0
    opt.device = torch.device("cuda:{}".format(opt.gpu_id))
    
    opt.weight_decay=5e-4
    # opt.GAUS=0.05 # fmnist -> 0.1
    # opt.loss2_alp=0.5
    opt.num_epoches = 40
    opt.lr = 0.0005
    opt.lr_step = 10
    opt.lr_gamma = 0.8

    opt.pgd_eps = 0.008
    opt.pgd_eps_iter = 0.0002
    opt.pgd_iterations = 1000
    opt.fgsm_eps = 0.015
    opt.attack = 'CW-U'
    opt.summary_name = '../summary/iforest'

    opt.classifier_net = 'vgg11bn'
    # opt.device = torch.device("cuda:{}".format(opt.gpu_id))
    opt.interpret_method = 'VG'
    
    opt.detect_method = 'iforest' # ['iforest','SVDD','LOF','Envelope'] # Envelope 太难跑
    opt.minus = False
    opt.scale = 1
    report_args(opt)

    for opt.classifier_net,opt.dataset,opt.image_channels,attack_list in [
        # ('vgg11bn','cifar10',3,[("FGSM-U",0.05),('DFool-U',0.05)]),
        # ('wide_resnet','imagenet',3,[("FGSM-U",0.05),('DFool-U',0.05)]),
        # ('vgg11bn','cifar10',3,[("PGD-T",0.007),("PGD-U",0.007),("DDN-U",1000),("DDN-T",1000),("CW-T",15),("CW-U",15),("NES-U",0),("Optim-U",0),("Boundary-U",0)]),
        # ('cwnet','cifar10',3,[("PGD-T",0.007),("PGD-U",0.007),("CW-T",9),("CW-U",9)]),
        # ('wide_resnet','imagenet',3,[("PGD-T",0.007),("PGD-U",0.007),("DDN-U",1000),("DDN-T",1000),("CW-T",9),("CW-U",9),("NES-U",0),("Optim-U",0),("Boundary-U",0)]),
        ('wide_resnet_small','imagenet',3,[('PGD-T',0.007),('PGD-U',0.007),('CW-T',9),('CW-U',9)]),
    ]:
        if 'mnist' in opt.dataset:
            opt.image_channels = 1
            opt.train_batchsize = opt.val_batchsize = 64
            opt.classifier_classes=10
            opt.workers=4
        elif "image" in opt.dataset:
            opt.train_batchsize = opt.val_batchsize = 4
            opt.classifier_classes=30
            opt.workers=2
        else:
            opt.train_batchsize = opt.val_batchsize = 128
            opt.classifier_classes=10
            opt.workers=4
        opt.tmp_dataset = f'tmp_dataset_{opt.classifier_net}_{opt.dataset}'
        auc_values = {}
        for opt.attack,opt.attack_param in attack_list:
            auc = main(isdiff=False)
            auc_d = main(isdiff=True)
            auc_values[f'{opt.attack}-{opt.attack_param}'] = {'advJudge':auc,'advJudge-R':1-auc,'advJudge-D':auc_d,'advJudge-D-R':1-auc_d}
        print(auc_values)
        timeStamp = datetime.now()
        formatTime = timeStamp.strftime("%m-%d %H-%M-%S")
        auc_values = pd.DataFrame.from_dict(auc_values)
        auc_values.to_csv(f'../{opt.tmp_dataset}/attack_eps/advJudge-{opt.classifier_net}-{opt.dataset}-{formatTime}.csv',float_format = '%.3f')



    ## recover
    # for opt.classifier_net,opt.dataset,opt.image_channels,attack_list in [
    #     ('vgg11bn','cifar10',3,[("PGD-T",0.007),("PGD-U",0.007),("DDN-U",1000),("DDN-T",1000),("CW-T",1),("CW-U",1),("NES-U",0),("Optim-U",0),("Boundary-U",0)]),
    #     # ('cwnet','cifar10',3,[("PGD-T",0.007),("PGD-U",0.007),("CW-T",9),("CW-U",9)]),
    #     # ('wide_resnet','imagenet',3,[("PGD-T",0.007),("PGD-U",0.007),("DDN-U",1000),("DDN-T",1000),("CW-T",9),("CW-U",9),("NES-U",0),("Optim-U",0),("Boundary-U",0)]),
    #     # ('wide_resnet_small','imagenet',3,[('PGD-T',0.007),('PGD-U',0.007),('CW-T',9),('CW-U',9)]),
    # ]:
    #     if 'mnist' in opt.dataset:
    #         opt.image_channels = 1
    #         opt.train_batchsize = opt.val_batchsize = 64
    #         opt.classifier_classes=10
    #         opt.workers=4
    #     elif "image" in opt.dataset:
    #         opt.train_batchsize = opt.val_batchsize = 4
    #         opt.classifier_classes=30
    #         opt.workers=2
    #     else:
    #         opt.train_batchsize = opt.val_batchsize = 128
    #         opt.classifier_classes=10
    #         opt.workers=4
    #     opt.tmp_dataset = f'tmp_dataset_{opt.classifier_net}_{opt.dataset}'
    #     auc_values = {}
    #     for opt.attack,opt.attack_param in attack_list:
    #         auc_values[f'{opt.attack}-{opt.attack_param}'] = recover()
    #     print(auc_values)
    #     timeStamp = datetime.now()
    #     formatTime = timeStamp.strftime("%m-%d %H-%M-%S")
    #     auc_values = pd.DataFrame.from_dict(auc_values)
    #     auc_values.to_csv(f'../{opt.tmp_dataset}/attack_eps/advJudge-recover-{formatTime}.csv',float_format = '%.3f')