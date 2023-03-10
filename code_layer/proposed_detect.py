import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from Options import Options,report_args,get_opt
import os
from datasets import build_loader,pairs_loader
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
from attack_methods import adversarialattack,blackboxAttacker
import re,math
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from interpreter_methods import interpretermethod
from detectors.XEnsemble import *
warnings.filterwarnings("ignore")

import pickle
def pk_dump(data,filename):
    with open(filename,'wb') as f:
        pickle.dump(data, f)
def pk_load(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    return data

@torch.enable_grad()
def getAdv(classifier,data, label):
    if opt.attack == 'Data':
        return data
    data=data.detach().clone()
    data.requires_grad=True
    clean_pred = classifier(data).max(1)[1]
    targeted = opt.attack.endswith('T')
    if targeted:
        targeted_label = []
        # Randomly choose targeted label that is not the ground truth one
        for i in range(data.size(0)):
            targeted_label.append(randint(1, opt.classifier_classes-1))
        attack_label = torch.fmod(label + torch.tensor(targeted_label).long().to(opt.device), opt.classifier_classes)
    else:
        # Untargeted attacks use the model classification labels
        attack_label = label

    # perturbed_data = white_cw_attack(classifier, data, attack_label,opt.classifier_classes,
    #                 targeted=targeted, learning_rate=opt.cw_lr, max_iterations=opt.cw_max_iterations,
    #                 confidence=opt.cw_confidence,attack_type=opt.loss_type)
    perturbed_data = adversarialattack(opt.attack, classifier, data, attack_label, opt)
    return perturbed_data

def get_aug(data,eps):
    return data + torch.randn(data.size()).cuda(device=opt.device) * eps

from torch.utils.data import Dataset, DataLoader, TensorDataset
def generate_interpret_only():
    if opt.interpret_method == 'org':
        return
    writer = SummaryWriter(opt.summary_name)
    # Initialize the network
    # choose classifier_net in package:models
    # classifier_net = 'vgg11bn'
    if 'imagenet' in opt.dataset:
        classifier_net = 'wide_resnet'
    else:
        classifier_net = 'vgg11bn'
    if 'black' in opt.attack_box :
        classifier_net = opt.classifier_net
    saved_name = f'../classifier_pth/classifier_{classifier_net}_{opt.dataset}_best.pth'
    model = eval(classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,dataset=opt.dataset)
    print('using network {}'.format(classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
        
    CleanInterpret = []
    AdvInterpret = []
    
    for d_name,i_name in [('AdvDataSet','AdvInterpret'),('CleanDataSet','CleanInterpret')]:
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
            writer.add_scalar(d_name+'correct',torch.sum(init_pred == target.data).item()/batch_size,global_step=global_step)
            global_step+=1

            interpreter = interpretermethod(model, opt.interpret_method)
            saliency_images = interpreter.interpret(data)
            interpreter.release()
            eval(i_name).append(saliency_images.clone().cpu())
        print("{}_correct = {:.4f}".format(d_name,correct.item()/size))

    try:
        os.makedirs(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}')
    except Exception as e:
        print(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}  already exist')

    try:
        os.makedirs(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}')
    except Exception as e:
        print(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}  already exist')

    for v in ['CleanInterpret','AdvInterpret']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/{v}.npy')

    return

def generate_result_only():
    writer = SummaryWriter(opt.summary_name)
    # Initialize the network
    # choose classifier_net in package:models
    # classifier_net = 'vgg11bn'
    if 'imagenet' in opt.dataset:
        classifier_net = 'wide_resnet'
    else:
        classifier_net = 'vgg11bn'
    if 'black' in opt.attack_box :
        classifier_net = opt.classifier_net
    saved_name = f'../classifier_pth/classifier_{classifier_net}_{opt.dataset}_best.pth'
    model = eval(classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,dataset=opt.dataset)
    print('using network {}'.format(classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

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
            writer.add_scalar(d_name+'correct',torch.sum(init_pred == target.data).item()/batch_size,global_step=global_step)
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
    writer = SummaryWriter(opt.summary_name)
    if opt.data_phase == 'test':
        loader = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=False, workers=opt.workers,shuffle=True)
    else:
        loader = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=True, workers=opt.workers,shuffle=True)

    # Initialize the network
    # choose classifier_net in package:models
    if 'imagenet' in opt.dataset:
        classifier_net = 'wide_resnet'
    else:
        classifier_net = 'vgg11bn'
    if 'black' in opt.attack_box :
        classifier_net = opt.classifier_net
    saved_name = f'../classifier_pth/classifier_{classifier_net}_{opt.dataset}_best.pth'
    model = eval(classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,dataset=opt.dataset)
    print('using network {}'.format(classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)

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
        # writer.add_scalar('adv_correct',torch.sum(init_pred == target.data).item()/batch_size,global_step=global_step)

        output = model(data)
        init_pred = output.max(1)[1]
        org_correct += torch.sum(init_pred==target.data)
        # writer.add_scalar('org_correct',torch.sum(init_pred == target.data).item()/batch_size,global_step=global_step)

        global_step+=1

        # Get data
        CleanDataSet.append(data.clone().cpu())
        Labels.append(target.clone().cpu())
        AdvDataSet.append(adv_data.clone().cpu())

        num += batch_size
        if org_correct.item()>opt.max_num:
            break

        
    print("num = {} , adv_correct = {:.4f} , org_correct = {:.4f}".format(num,correct.item()/num,org_correct.item()/num))

    try:
        os.makedirs(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}')
    except Exception as e:
        print(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}  already exist')

    for v in ['CleanDataSet','AdvDataSet','Labels']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')

    return


def generate_black():
    writer = SummaryWriter(opt.summary_name)
    if opt.data_phase == 'test':
        loader = pairs_loader(f'../data/{opt.dataset}/{opt.data_phase}_pair.pth', opt.train_batchsize,workers=opt.workers,shuffle=True)
    else:
        loader = pairs_loader(f'../data/{opt.dataset}/{opt.data_phase}_pair.pth', opt.train_batchsize,workers=opt.workers,shuffle=False)

    # Initialize the network
    # choose classifier_net in package:models
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,dataset=opt.dataset)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()
    
    attacker = blackboxAttacker(opt.attack,model,opt)

    ## attack
    correct = 0
    org_correct = 0
    global_step = 0
    size = len(loader.dataset)
    print('dataset size is ',size)
    
    CleanDataSet = []
    AdvDataSet = []
    Labels = []

    num = 0
    _pnt = True
    # Loop over all examples in test set
    for data, target,pair,ptarget in tqdm(loader, desc=opt.data_phase+"_"+opt.attack):
        # Send the data and label to the device
        data, target = data.to(opt.device), target.to(opt.device)
        pair,ptarget = pair.to(opt.device), ptarget.to(opt.device)
        batch_size = data.shape[0]
        if _pnt:
            print(batch_size)
            _pnt=False
        adv_data = attacker(data,target,pair,ptarget)

        # Forward pass the data through the model
        output = model(adv_data)
        init_pred = output.max(1)[1] # get the index of the max log-probability
        correct += torch.sum(init_pred == target.data)
        writer.add_scalar('adv_correct',torch.sum(init_pred == target.data).item()/batch_size,global_step=global_step)

        output = model(data)
        init_pred = output.max(1)[1]
        org_correct += torch.sum(init_pred==target.data)
        writer.add_scalar('org_correct',torch.sum(init_pred == target.data).item()/batch_size,global_step=global_step)

        global_step+=1

        # Get data
        CleanDataSet.append(data.clone().cpu())
        Labels.append(target.clone().cpu())
        AdvDataSet.append(adv_data.clone().cpu())

        num+=batch_size
        if num>opt.max_num:
            break

        
    print("num = {} , adv_correct = {:.4f} , org_correct = {:.4f}".format(num,correct.item()/num,org_correct.item()/num))

    try:
        os.makedirs(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}')
    except Exception as e:
        print(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}  already exist')

    for v in ['CleanDataSet','AdvDataSet','Labels']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')

    return

def save_roc_cruve(y,prob,path):
    try:
        os.makedirs(path)
    except:
        pass
    pk_dump(y,os.path.join(path,'y'))
    pk_dump(prob,os.path.join(path,'prob'))

import sklearn
def save_roc_fig(y,prob,path,title = 'Receiver operating characteristic example'):
    fpr,tpr,thresholds = roc_curve(y,prob) ###计算真正率和假正率
    roc_auc = sklearn.metrics.auc(fpr,tpr) ###计算auc的值
    threshold,point = Find_Optimal_Cutoff(tpr,fpr,thresholds)
    
    lw = 2
    # plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    plt.plot(point[0],point[1],'ro')
    plt.savefig(path)
    plt.close()

def redraw():
    Z = pk_load(f'../roc_cruve_data/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/PCA-{opt.detect_method}/Normal/prob')
    Y = pk_load(f'../roc_cruve_data/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/PCA-{opt.detect_method}/Normal/y')
    save_roc_fig(Y,Z,f'../roc_cruve_data/figs/pca/{opt.data_phase}_{opt.attack}_{opt.attack_param}_{opt.interpret_method}_PCA-{opt.detect_method}_Normal.png',
        title = f'{opt.data_phase}_{opt.attack}_{opt.attack_param}_{opt.interpret_method}_PCA-{opt.detect_method}_AUC曲线')

    Z = pk_load(f'../roc_cruve_data/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/PCA-{opt.detect_method}/Reverse/prob')
    Y = pk_load(f'../roc_cruve_data/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/PCA-{opt.detect_method}/Reverse/y')
    save_roc_fig(Y,Z,f'../roc_cruve_data/figs/pca/{opt.data_phase}_{opt.attack}_{opt.attack_param}_{opt.interpret_method}_PCA-{opt.detect_method}_Reverse.png',
        title = f'{opt.data_phase}_{opt.attack}_{opt.attack_param}_{opt.interpret_method}_PCA-{opt.detect_method}_AUC曲线')

def get_sift_data():
    print("generate_result_only")
    generate_result_only()
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    if opt.interpret_method=='org':
        CleanInterpret = CleanDataSet
        AdvInterpret = AdvDataSet
    else:
        try:
            CleanInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
            AdvInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')
        except:
            generate_interpret_only()
            CleanInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
            AdvInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')

    CleanResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanResult.npy')
    AdvResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvResult.npy')

    CleanDataSet = CleanDataSet[CleanResult==Labels]
    AdvDataSet = AdvDataSet[(CleanResult==Labels)&(AdvResult!=Labels)]
    CleanInterpret = CleanInterpret[CleanResult==Labels]
    AdvInterpret = AdvInterpret[(CleanResult==Labels)&(AdvResult!=Labels)]
    return CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels

def get_difference(p):
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    x = CleanDataSet-AdvDataSet
    x = x.view(x.shape[0],-1)
    l = torch.norm(x,dim=1,p=p).mean().item()/x.shape[1]
    return l

class Normalizer:
    def __init__(self):
        self.mean = None 
        self.max = None 
        self.min = None 
        self.std = None 
    
    def fit(self,x):
        self.max = np.max(x)
        self.min = np.min(x)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
    
    def get(self,x,method = 'min-max'):
        if method == 'min-max':
            # print('norm with min-max')
            return (x-self.min)/(self.max-self.min)
        else:
            # print('norm with mean-std')
            return (x-self.mean)/self.std
    def dump(self,path):
        make_dir(path)
        pk_dump(self.min,os.path.join(path,'min.pth'))
        pk_dump(self.max,os.path.join(path,'max.pth'))
        pk_dump(self.mean,os.path.join(path,'mean.pth'))
        pk_dump(self.std,os.path.join(path,'std.pth'))
    
    def load(self,path):
        self.max = pk_load(os.path.join(path,'max.pth'))
        self.min = pk_load(os.path.join(path,'min.pth'))
        self.mean = pk_load(os.path.join(path,'mean.pth'))
        self.std = pk_load(os.path.join(path,'std.pth'))

def auc_curve(y,prob,plt = True):
    fpr,tpr,thresholds = roc_curve(y,prob) ###计算真正率和假正率
    roc_auc = sklearn.metrics.auc(fpr,tpr) ###计算auc的值
    threshold,point = Find_Optimal_Cutoff(tpr,fpr,thresholds)
    
    if plt:
        lw = 2
        # plt.figure(figsize=(10,10))
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
    
        
        plt.plot(point[0],point[1],'ro')

        plt.show()

    return roc_auc,threshold,thresholds

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
    CleanInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
    AdvInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')
    size = CleanInterpret.shape[0]

    for v in ['CleanDataSet','AdvDataSet','CleanInterpret','AdvInterpret']:
        n = eval(v)
        try:
            os.makedirs(f'../maps/{opt.interpret_method}/{opt.attack}_{opt.attack_param}')
        except Exception as e:
            print(f'../maps/{opt.interpret_method}/{opt.attack}_{opt.attack_param}  already exist')

        vutils.save_image(n[:64],f'../maps/{opt.interpret_method}/{opt.attack}_{opt.attack_param}/{v}.jpg',nrow=8)
    
    x = CleanDataSet-AdvDataSet
    x = x.view(size,-1)
    l2 = torch.norm(x,dim=1,p=2).mean().item()
    print('l2 distance is {:.4f}'.format(l2))

def make_dir(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(f'{path}  already exist')

def plot_auc_eps(data,path,methods=['LRP','org','VG','GBP','IG','attack_rate']):
    for method in methods:
        x =[]
        y =[]
        for eps,item in data.items():
            x.append(eps)
            y.append(item[method])
        plt.plot(x,y,label=method)
    plt.xlabel('eps')
    plt.ylabel('auc')
    plt.title(path)
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.close()


def get_difference(p):
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    x = CleanDataSet-AdvDataSet
    x = x.view(x.shape[0],-1)
    l = torch.norm(x,dim=1,p=p).mean().item()
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
    elif "Boundary" in opt.attack:
        generate_black()
    elif "NES" in opt.attack:
        generate_black()
    elif "Optim" in opt.attack:
        generate_black()
    
    generate_result_only()
    for opt.interpret_method in opt.interpret_methods:
        generate_interpret_only()
    CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
    attack_rate = AdvInterpret.shape[0]/CleanInterpret.shape[0]
    l2 = get_difference(2)
    print('attack is {}, param is {:.4f}, attack rate is {:.4f}, l2 is {:.4f}'.format(
        opt.attack,opt.attack_param,attack_rate,l2
    ))

def train_classifier(load=False):
    loader = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=True, workers=opt.workers,shuffle=True)

    # Initialize the network
    # choose classifier_net in package:models
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,dataset=opt.dataset)
    print('using network {}'.format(opt.classifier_net))
    if load:
        print('loading from {}'.format(saved_name))
        model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
    criterion = nn.CrossEntropyLoss()

    model = model.to(opt.device)
    model.train()
    _pnt = True
    for epoch in range(opt.num_epoches):
        print('Epoch {}/{}'.format(epoch+1, opt.num_epoches))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data
        size = len(loader.dataset)
        
        for inputs, labels in tqdm(loader, desc=f'{opt.dataset}_{opt.classifier_net}'):
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if _pnt:
                print("inputs shape {}".format(inputs.shape))
                print("labels shape {}".format(labels.shape))
                print("outputs shape {}".format(outputs.shape))
                _pnt = False
            preds = outputs.max(1)[1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / size
        epoch_acc = running_corrects.double() / size
        scheduler.step()
        
        print('train Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))
        torch.save(model.state_dict(), saved_name)
        if(epoch_acc > 0.85):
            break
    return model

def test_classifier():
    loader = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=False, workers=opt.workers,shuffle=True)

    # Initialize the network
    # choose classifier_net in package:models
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,dataset=opt.dataset)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    running_corrects = 0
    # Iterate over data
    size = len(loader.dataset)
    
    for inputs, labels in tqdm(loader, desc=f'{opt.dataset}_{opt.classifier_net}'):
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)

        outputs = model(inputs)
        preds = outputs.max(1)[1]

        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / size
    
    print('train Acc: {:.4f}'.format(epoch_acc))

    return model


from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import sklearn
def main(interpret_methods=['DL','LRP','org','VG','GBP','IG'],reverse_list=[],train = True):
    if 'black' in opt.attack_box and 'imagenet' in opt.dataset:
        reducer = eval(opt.reducer)(n_components=opt.max_num-10)
    else:
        reducer = eval(opt.reducer)()
    ensembled_clf = Ensembler(interpret_methods=interpret_methods,detect_method=opt.detect_method,use_voting=opt.use_voting)
    clf_path_name = f'../classifier_pth/{opt.classifier_net}_{opt.dataset}_{opt.detect_method}_clf-ensemble'

    reverse = {}
    for i in interpret_methods:
        reverse[i] = False
    for i in reverse_list:
        reverse[i] = True

    for opt.interpret_method in interpret_methods:
        generate_interpret_only()
        CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
        cleansize = CleanInterpret.shape[0]
        advsize = AdvInterpret.shape[0]
        total = Labels.shape[0]
        print(f"i_method is {opt.interpret_method}, total is {total}, clean size is {cleansize} (clean/total={cleansize/total}), adv size is {advsize} (adv/clean={advsize/cleansize})")

        if train:
            print('training begin')
            t1 = datetime.now()
            X_train = reducer.train(CleanInterpret)
            reducer.dump(os.path.join(clf_path_name,f"reducer_{opt.data_phase}_{opt.interpret_method}.npy"))
            ensembled_clf.fit(opt.interpret_method,X_train)
            t2 = datetime.now()
            totalseconds = (t2-t1).total_seconds()
            h = totalseconds // 3600
            m = (totalseconds - h*3600) // 60
            s = totalseconds - h*3600 - m*60
            print('training end in time {}h {}m {:.4f}s'.format(h,m,s))
            ensembled_clf.save(clf_path_name)
        else:
            # if not os.path.exists(os.path.join(clf_path_name,f"reducer_{opt.data_phase}_{opt.interpret_method}.npy")):
                # print(f"train reducer {opt.data_phase}")
                # X_train = reducer.train(CleanInterpret)
                # reducer.dump(os.path.join(clf_path_name,f"reducer_{opt.data_phase}_{opt.interpret_method}.npy"))
            # reducer.load(os.path.join(clf_path_name,f"reducer_{opt.data_phase}_{opt.interpret_method}.npy"))
            # X_train = reducer.reduce(CleanInterpret)
            X_train = reducer.train(CleanInterpret)
            ensembled_clf.load(clf_path_name)

        print('eval begin')
        t1 = datetime.now()
        InterpretAll = torch.cat([CleanInterpret,AdvInterpret],dim=0)
        org_size = CleanInterpret.shape[0]
        adv_size = AdvInterpret.shape[0]
        InterpretAll = reducer.reduce(InterpretAll)
        if 'cifar' in opt.dataset:
            norm_method = 'min-max'
        else:
            norm_method = 'mean-std'

        # if not train:
        #     make_dir(f"../{opt.tmp_dataset}/figs/{opt.data_phase}/{opt.attack}-{opt.attack_param}")
        #     img_path = f"../{opt.tmp_dataset}/figs/{opt.data_phase}/{opt.attack}-{opt.attack_param}/{opt.interpret_method}_dist.png"
        #     dist = ensembled_clf.draw_dist_distance(InterpretAll[:org_size],InterpretAll[org_size:],img_path)
        #     img_path = f"../{opt.tmp_dataset}/figs/{opt.data_phase}/{opt.attack}-{opt.attack_param}/{opt.interpret_method}_auc_all.png"
        #     Y = torch.cat([torch.ones(org_size),torch.zeros(adv_size)],dim=0).numpy()
        #     aucs = ensembled_clf.draw_all_auc(InterpretAll,Y,img_path)
        #     col = np.argmax(aucs)
        #     col2 = np.argmax(dist)
        #     img_path = f"../{opt.tmp_dataset}/figs/{opt.data_phase}/{opt.attack}-{opt.attack_param}/{opt.interpret_method}_hist-{col}.png"
        #     ensembled_clf.draw_hist_col(InterpretAll[:org_size],InterpretAll[org_size:],col,img_path)
        #     img_path = f"../{opt.tmp_dataset}/figs/{opt.data_phase}/{opt.attack}-{opt.attack_param}/{opt.interpret_method}_point-{col}-{col2}.png"
        #     ensembled_clf.draw_point(InterpretAll[:org_size],InterpretAll[org_size:],col,col2,img_path)

        ensembled_clf.calculate(opt.interpret_method,InterpretAll,norm_method=norm_method,reverse=reverse[opt.interpret_method])
        t2 = datetime.now()
        totalseconds = (t2-t1).total_seconds()
        h = totalseconds // 3600
        m = (totalseconds - h*3600) // 60
        s = totalseconds - h*3600 - m*60
        print('eval end in time {}h {}m {:.4f}s'.format(h,m,s))

    # timeStamp = datetime.now()
    # formatTime = timeStamp.strftime("%m-%d %H-%M-%S")
    # ensembled_clf.save_z(torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy(),f'../{opt.tmp_dataset}/attack_eps/{opt.dataset}_{opt.classifier_net}_Z--{formatTime}.csv')
    # exit()

    # ensembled_clf.fit_forest_ensembler()
    # ensembled_clf.save(clf_path_name)

    Z = ensembled_clf.ensemble(method="min")
    Y = torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy()
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]>=threshold).sum() + (Z[cleansize:]<threshold).sum()
    acc = acc/(cleansize+advsize)
    print('{}_{} :: X-Det :: Normal :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(opt.dataset,opt.classifier_net,roc_auc,threshold,acc))
    roc_auc_rev= roc_auc

    Z = ensembled_clf.ensemble(method='max')
    Y = torch.cat([torch.zeros(cleansize),torch.ones(advsize)],dim=0).numpy()
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]<threshold).sum() + (Z[cleansize:]>=threshold).sum()
    acc = acc/(cleansize+advsize)
    print('{}_{} :: X-Det :: Reverse :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(opt.dataset,opt.classifier_net,roc_auc,threshold,acc))

    if roc_auc>roc_auc_rev:
        auc_min = roc_auc
    else:
        auc_min = roc_auc_rev

    item = {}
    item['X-Det'] = auc_min
    CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
    item['attack_rate'] = AdvInterpret.shape[0]/CleanInterpret.shape[0]
    item['l2'] = get_difference(2)

    for i_method in interpret_methods:
        Z = ensembled_clf.sub_detector(i_method)

        Y = torch.cat([torch.zeros(cleansize),torch.ones(advsize)],dim=0).numpy()
        roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
        acc = (Z[:cleansize]<threshold).sum() + (Z[cleansize:]>=threshold).sum()
        acc = acc/(cleansize+advsize)
        print('{}_{} :: {} :: Reverse :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(opt.dataset,opt.classifier_net,i_method,roc_auc,threshold,acc))
        roc_auc_rev= roc_auc

        Y = torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy()  # 反向操作
        roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
        acc = (Z[:cleansize]>=threshold).sum() + (Z[cleansize:]<threshold).sum()
        acc = acc/(cleansize+advsize)
        print('{}_{} :: {} :: Normal :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(opt.dataset,opt.classifier_net,i_method,roc_auc,threshold,acc))

        item[i_method] = max(roc_auc,roc_auc_rev)

    return item


def draw_pca_reduce_map(interpret_methods=['DL','LRP','org','VG','GBP','IG']):
    reducer = eval(opt.reducer)()
    # ensembled_clf = Ensembler(interpret_methods=interpret_methods,detect_method='iforest')
    # clf_path_name = f'../classifier_pth/{opt.classifier_net}_{opt.dataset}_{opt.detect_method}_clf-ensemble'
    for opt.interpret_method in interpret_methods:
        generate_interpret_only()
        CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
        cleansize = CleanInterpret.shape[0]
        advsize = AdvInterpret.shape[0]
        total = Labels.shape[0]
        print(f"i_method is {opt.interpret_method}, total is {total}, clean size is {cleansize} (clean/total={cleansize/total}), adv size is {advsize} (adv/clean={advsize/cleansize})")
        X_train = reducer.train(CleanInterpret)

        InterpretAll = torch.cat([CleanInterpret,AdvInterpret],dim=0)
        InterpretAll = reducer.reduce(InterpretAll)
        make_dir(f"../{opt.tmp_dataset}/figs/{opt.data_phase}/{opt.attack}-{opt.attack_param}")
        for x,y in [(0,1),(-2,-1)]:
            plt.figure()
            X_clean = InterpretAll[:cleansize,x]
            Y_clean = InterpretAll[:cleansize,y]
            plt.plot(X_clean,Y_clean,'ro')
            X_adv = InterpretAll[cleansize:,x]
            Y_adv = InterpretAll[cleansize:,y]
            plt.plot(X_adv,Y_adv,'bx')
            plt.savefig(f"../{opt.tmp_dataset}/figs/{opt.data_phase}/{opt.attack}-{opt.attack_param}/{opt.interpret_method}_{x}_{y}.png")
            plt.close()

    return



if __name__ == '__main__':
    print("proposed_detect.py")
    opt = Options().parse_arguments()
    t1 = datetime.now()
    opt.dataset = 'imagenet'
    # opt.dataset = 'fmnist'
    opt.gpu_id = 0
    opt.device = torch.device("cuda:{}".format(opt.gpu_id))
    
    opt.weight_decay=5e-4
    # opt.GAUS=0.05 # fmnist -> 0.1
    # opt.loss2_alp=0.5
    opt.num_epoches = 100
    opt.lr = 0.01
    opt.lr_step = 10
    opt.lr_gamma = 0.8

    opt.pgd_eps = 0.08
    opt.pgd_eps_iter = 0.0005
    opt.pgd_iterations = 500
    opt.fgsm_eps = 0.015
    opt.attack = 'CW-U'
    opt.summary_name = '../summary/iforest'

    if 'mnist' in opt.dataset:
        opt.image_channels = 1
        opt.train_batchsize = opt.val_batchsize = 4
        # opt.classifier_net = 'vgg11bn'
        opt.workers=4
    elif "image" in opt.dataset:
        opt.train_batchsize = opt.val_batchsize = 4
        # opt.classifier_net='resnet30'
        opt.workers=2
    else:
        opt.train_batchsize = opt.val_batchsize = 128
        # opt.classifier_net = 'vgg11bn'
        opt.workers=4

    # opt.classifier_net = 'vgg11bn'
    opt.classifier_net = 'cwnet'
    # opt.device = torch.device("cuda:{}".format(opt.gpu_id))
    opt.interpret_method = 'VG'
    opt.data_phase = 'test'
    opt.detect_method = 'iforest' # ['iforest','SVDD','LOF','Envelope'] # Envelope 太难跑
    opt.minus = False
    opt.scale = 1
    opt.use_voting=False


    opt.classifier_net='wide_resnet_small'
    opt.tmp_dataset = f'tmp_dataset_{opt.classifier_net}_{opt.dataset}'
    # opt.dataset = 'imagenet'
    opt.image_channels=3
    opt.train_batchsize = opt.val_batchsize = 64
    opt.dfool_max_iterations = 600
    # opt.reducer = PCA_Reducer
    report_args(opt)


    # if "imagenet" in opt.dataset:
    #     opt.classifier_classes=30
    # train_classifier()
    # test_classifier()


    opt.optim_iter=50
    opt.optim_alpha=0.2 # 二分查找的初始倍率
    opt.optim_beta=0.05 # 每次旋转的角度
    opt.filter_g = 0.8
    opt.stop_g = 0.5
    opt.max_num=1000
    opt.interpret_methods = ['VG','IG','GBP',"DL","org","LRP"]
    opt.reverse_list=[]
    opt.attack_box='white'
    # opt.interpret_methods = ['VG']
    # opt.interpret_methods = ['VG','IG',"DL"]
    # for opt.classifier_net,opt.dataset,opt.image_channels in [('vgg11bn','fmnist',1),('cwnet','fmnist',1),('cwnet','cifar10',3)]:
    # for opt.classifier_net,opt.dataset,opt.image_channels in [('vgg11bn','cifar10',3)]:
    for opt.classifier_net,opt.dataset,opt.image_channels,opt.attack_box,attack_list in [
        ('vgg11bn','cifar10',3,'white',[("PGD-T",0.015),("PGD-U",0.007),("FGSM-U",0.05),('DFool-U',0.05),("DDN-U",1000),("DDN-T",1000),("CW-T",15),("CW-U",15)]),
        ('vgg11bn','cifar10',3,'black',[("NES-U",0),("Optim-U",0),("Boundary-U",0)]),
        ('cwnet','cifar10',3,'trans',[("PGD-T",0.007),("PGD-U",0.007),("CW-T",9),("CW-U",9)]),
        ('wide_resnet','imagenet',3,'white',[("FGSM-U",0.05),('DFool-U',0.05),("PGD-T",0.007),("PGD-U",0.007),("DDN-U",1000),("DDN-T",1000),("CW-T",9),("CW-U",9)]),
        ('wide_resnet','imagenet',3,'black',[("NES-U",0),("Optim-U",0),("Boundary-U",0)]),
        ('wide_resnet_small','imagenet',3,'trans',[('PGD-T',0.007),('PGD-U',0.007),('CW-T',9),('CW-U',9)]),
    ]:
        if "imagenet" in opt.dataset:
            opt.classifier_classes=30
            if "LRP" in opt.interpret_methods:
                opt.interpret_methods.remove("LRP")
            opt.detect_method = "iforest"
            # opt.use_voting=True
            opt.reverse_list=["GBP"]
            opt.reducer = "FFT_PCA_Reducer"
            opt.train_batchsize = opt.val_batchsize = 32
        if "cifar" in opt.dataset:
            opt.train_batchsize = opt.val_batchsize = 128
            opt.detect_method="iforest"
            opt.reducer = "PCA_Reducer"
        opt.tmp_dataset = f'tmp_dataset_{opt.classifier_net}_{opt.dataset}'
        auc_values = {}
        auc_values_train = {}
        train=False
        # train=True
        if "black" in opt.attack_box:
            opt.max_num=500
            attack_list = [("Optim-U",0),("NES-U",0),("Boundary-U",0)] 
        for opt.attack,opt.attack_param in attack_list:
            # train dataset
            opt.data_phase='train'
            try:
                _,_,_,_,_ = get_sift_data()
            except Exception as e:
                print(e)
                print(f"{opt.dataset}_{opt.classifier_net}_{opt.attack}_{opt.attack_param}_{opt.data_phase} initial generate")
                generate_all()
            # draw_pca_reduce_map(interpret_methods=opt.interpret_methods)
            item = main(train=train,interpret_methods=opt.interpret_methods,reverse_list=opt.reverse_list)
            auc_values_train[f"{opt.attack}_{opt.attack_param}"] = item
            train=False

            # test dataset
            opt.data_phase='test'
            try:
                _,_,_,_,_ = get_sift_data()
            except Exception as e:
                print(e)
                print(f"{opt.dataset}_{opt.classifier_net}_{opt.attack}_{opt.attack_param}_{opt.data_phase} initial generate")
                generate_all()
            # draw_pca_reduce_map(interpret_methods=opt.interpret_methods)
            item = main(train=False,interpret_methods=opt.interpret_methods,reverse_list=opt.reverse_list)
            auc_values[f"{opt.attack}_{opt.attack_param}"] = item
        print(auc_values_train)
        auc_values_train = pd.DataFrame.from_dict(auc_values_train)
        timeStamp = datetime.now()
        formatTime = timeStamp.strftime("%m-%d %H-%M-%S")
        auc_values_train.to_csv(f'../{opt.tmp_dataset}/attack_eps/{opt.dataset}_{opt.classifier_net}_X-Det_train--{formatTime}.csv',float_format = '%.3f')
        print(auc_values)
        auc_dict = pd.DataFrame.from_dict(auc_values)
        auc_dict.to_csv(f'../{opt.tmp_dataset}/attack_eps/{opt.dataset}_{opt.classifier_net}_X-Det_test--{formatTime}.csv',float_format = '%.3f')
            
    # for opt.classifier_net,opt.dataset,opt.image_channels in [('cwnet','cifar10',3)]:
    #     opt.tmp_dataset = f'tmp_dataset_{opt.classifier_net}_{opt.dataset}'
    #     auc_values = {}
    #     auc_values_train = {}
    #     train=True
    #     for opt.attack,opt.attack_param in [ ('PGD-T',0.007),('PGD-U',0.007),('CW-T',9),('CW-U',9)]:
    #         # train dataset
    #         opt.data_phase='train'
    #         try:
    #             _,_,_,_,_ = get_sift_data()
    #         except:
    #             print(f"{opt.dataset}_{opt.classifier_net}_{opt.attack}_{opt.attack_param}_{opt.data_phase} initial generate")
    #             generate_all()
    #         item = main(train=train)
    #         auc_values_train[opt.attack_param] = item
    #         train=False
    #         # test dataset
    #         opt.data_phase='test'
    #         try:
    #             _,_,_,_,_ = get_sift_data()
    #         except:
    #             print(f"{opt.dataset}_{opt.classifier_net}_{opt.attack}_{opt.attack_param}_{opt.data_phase} initial generate")
    #             generate_all()
    #         item = main(train=False)
    #         auc_values[opt.attack_param] = item
    #     print(auc_values_train)
    #     auc_values_train = pd.DataFrame.from_dict(auc_values_train)
    #     auc_values_train.to_csv(f'../{opt.tmp_dataset}/attack_eps/{opt.attack}_X-Det_train.csv',float_format = '%.3f')
    #     print(auc_values)
    #     auc_dict = pd.DataFrame.from_dict(auc_values)
    #     auc_dict.to_csv(f'../{opt.tmp_dataset}/attack_eps/{opt.attack}_X-Det_test.csv',float_format = '%.3f')

    # param_dic={
    #     'FGSM-U':[0.04,0.035,0.03,0.025,0.02,0.015,0.01,0.005],
    #     'PGD-T':[0.005,0.007,0.01,0.013,0.015],
    #     'PGD-U':[0.005,0.007,0.01,0.013,0.015],
    #     'CW-U':[1,3,6,9,15],
    #     'CW-T':[1,3,6,9,15],
    # }

    # opt.max_num = 10000
    # for opt.attack in ['FGSM-U','PGD-U','PGD-T','CW-U','CW-T']:
    # # for opt.attack in ['PGD-U','PGD-T','CW-U','CW-T']:
    #     auc_values = {}
    #     auc_values_train = {}
    #     train=False
    #     for opt.attack_param in param_dic[opt.attack]:
    #         # train dataset
    #         opt.data_phase='train'
    #         try:
    #             _,_,_,_,_ = get_sift_data()
    #         except:
    #             print(f"{opt.attack}_{opt.attack_param}_{opt.data_phase} initial generate")
    #             generate_all()
    #         item = main(train=train)
    #         auc_values_train[opt.attack_param] = item
    #         train=False
    #         # test dataset
    #         opt.data_phase='test'
    #         try:
    #             _,_,_,_,_ = get_sift_data()
    #         except:
    #             print(f"{opt.attack}_{opt.attack_param}_{opt.data_phase} initial generate")
    #             generate_all()
    #         item = main(train=False)
    #         auc_values[opt.attack_param] = item

    #     print(auc_values_train)
    #     auc_values_train = pd.DataFrame.from_dict(auc_values_train)
    #     auc_values_train.to_csv(f'../{opt.tmp_dataset}/attack_eps/{opt.attack}_X-Det_train.csv',float_format = '%.3f')
    #     print(auc_values)
    #     auc_dict = pd.DataFrame.from_dict(auc_values)
    #     auc_dict.to_csv(f'../{opt.tmp_dataset}/attack_eps/{opt.attack}_X-Det_test.csv',float_format = '%.3f')