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

def test(opt):
    writer = SummaryWriter(opt.summary_name)
    test_loader = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=False, workers=opt.workers)

    # Initialize the network
    # choose classifier_net in package:models
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_noise.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)

    ## attack
    correct = 0
    org_correct = 0
    global_step = 0
    size = len(test_loader.dataset)
    model.eval()
    print(size)

    # Loop over all examples in test set
    for data, target in tqdm(test_loader, desc='Test'):
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
        writer.add_scalar('adv_correct',torch.sum(init_pred == target.data).item()/batch_size,global_step=global_step)

        output = model(data)
        init_pred = output.max(1)[1]
        org_correct += torch.sum(init_pred==target.data)
        writer.add_scalar('org_correct',torch.sum(init_pred == target.data).item()/batch_size,global_step=global_step)

        global_step+=1
    print("adv_correct = {:.4f} , org_correct = {:.4f}".format(correct.item()/size,org_correct.item()/size))
    return

from torch.utils.data import Dataset, DataLoader, TensorDataset
def generate_interpret_only():
    if opt.interpret_method == 'org':
        return
    writer = SummaryWriter(opt.summary_name)
    # Initialize the network
    # choose classifier_net in package:models
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    CleanDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/AdvDataSet.npy')
    Labels = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/Labels.npy')
        
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
        os.makedirs(f'../tmp_dataset/{opt.data_phase}/{opt.attack}')
    except Exception as e:
        print(f'../tmp_dataset/{opt.data_phase}/{opt.attack}  already exist')

    try:
        os.makedirs(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{opt.interpret_method}')
    except Exception as e:
        print(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{opt.interpret_method}  already exist')

    for v in ['CleanInterpret','AdvInterpret']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/{v}.npy')

    return

def generate(inter=True):
    if opt.interpret_method == 'org':
        return
    writer = SummaryWriter(opt.summary_name)
    test_loader = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=False, workers=opt.workers)
    train_loader = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=True, workers=opt.workers)
    if opt.data_phase == 'test':
        loader = test_loader
    else:
        loader = train_loader

    # Initialize the network
    # choose classifier_net in package:models
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)

    evalmodel = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('eval using network {}'.format(opt.classifier_net))
    print('eval loading from {}'.format(saved_name))
    evalmodel.load_state_dict(torch.load(saved_name,map_location=opt.device))
    evalmodel = evalmodel.to(opt.device)
    evalmodel.eval()

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
        output = evalmodel(adv_data)
        init_pred = output.max(1)[1] # get the index of the max log-probability
        correct += torch.sum(init_pred == target.data)
        writer.add_scalar('adv_correct',torch.sum(init_pred == target.data).item()/batch_size,global_step=global_step)

        output = evalmodel(data)
        init_pred = output.max(1)[1]
        org_correct += torch.sum(init_pred==target.data)
        writer.add_scalar('org_correct',torch.sum(init_pred == target.data).item()/batch_size,global_step=global_step)

        global_step+=1

        # Get data
        CleanDataSet.append(data.clone().cpu())
        Labels.append(target.clone().cpu())
        AdvDataSet.append(adv_data.clone().cpu())

        if inter:
            interpreter = interpretermethod(model, opt.interpret_method)
            saliency_images = interpreter.interpret(data)
            interpreter.release()
            CleanInterpret.append(saliency_images.detach().clone().cpu())

            interpreter = interpretermethod(model, opt.interpret_method)
            saliency_images = interpreter.interpret(adv_data)
            interpreter.release()
            AdvInterpret.append(saliency_images.detach().clone().cpu())
        
    print("adv_correct = {:.4f} , org_correct = {:.4f}".format(correct.item()/size,org_correct.item()/size))

    try:
        os.makedirs(f'../tmp_dataset/{opt.data_phase}/{opt.attack}')
    except Exception as e:
        print(f'../tmp_dataset/{opt.data_phase}/{opt.attack}  already exist')

    try:
        os.makedirs(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{opt.interpret_method}')
    except Exception as e:
        print(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{opt.interpret_method}  already exist')
        
    for v in ['CleanDataSet','AdvDataSet','Labels']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{v}.npy')

    if inter:
        for v in ['CleanInterpret','AdvInterpret']:
            n = eval(v)
            n = torch.cat(n,dim=0)
            torch.save(n,f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/{v}.npy')

    return correct.item()/size

def save_roc_cruve(y,prob,path):
    try:
        os.makedirs(path)
    except:
        pass
    pk_dump(y,os.path.join(path,'y'))
    pk_dump(prob,os.path.join(path,'prob'))

def save_roc_fig(y,prob,path,title = 'Receiver operating characteristic example'):
    fpr,tpr,thresholds = roc_curve(y,prob) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
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

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
def main(train = True):
    CleanDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/AdvDataSet.npy')
    Labels = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/Labels.npy')
    if opt.interpret_method=='org':
        CleanInterpret = CleanDataSet
        AdvInterpret = AdvDataSet
    else:
        CleanInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/CleanInterpret.npy')
        AdvInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/AdvInterpret.npy')
    size = CleanInterpret.shape[0]

    # for v in ['CleanDataSet','AdvDataSet','CleanInterpret','AdvInterpret','Labels']:
    #     n = eval(v)
    #     print(f"{v} shape is : {n.shape}")

    clf_path_name = f'../classifier_pth/{opt.detect_method}_clf-{opt.interpret_method}.pkl'
    if train:
        print('training begin')
        t1 = datetime.now()
        rng = np.random.RandomState(420)
        X_train = CleanInterpret.view(size,-1).numpy()*opt.scale
        if opt.detect_method == 'LOF':
            clf = LocalOutlierFactor(n_neighbors=20,novelty=True)
        elif opt.detect_method == 'SVDD':
            clf = OneClassSVM(kernel='rbf', degree=3, gamma='auto',
                coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False,
                max_iter=-1)
        elif opt.detect_method == 'iforest':
            clf = IsolationForest(n_estimators=200,max_samples=0.8, max_features=0.5, random_state=rng,warm_start=True)
        elif opt.detect_method == 'Envelope':
            clf = EllipticEnvelope()
        else:
            raise Exception(f"{opt.detect_method} not implemented")
        clf.fit(X_train)
        t2 = datetime.now()
        totalseconds = (t2-t1).total_seconds()
        h = totalseconds // 3600
        m = (totalseconds - h*3600) // 60
        s = totalseconds - h*3600 - m*60
        print('training end in time {}h {}m {:.4f}s'.format(h,m,s))
        pk_dump(clf,clf_path_name)
    else:
        clf = pk_load(clf_path_name)

    print('eval begin')
    t1 = datetime.now()
    InterpretAll = torch.cat([CleanInterpret,AdvInterpret],dim=0).view(size*2,-1).numpy()*opt.scale
    Z = clf.decision_function(InterpretAll)
    if opt.minus:
        Z = -Z
    Y = torch.cat([torch.zeros(size),torch.ones(size)],dim=0).numpy()
    Y = Y
    t2 = datetime.now()
    totalseconds = (t2-t1).total_seconds()
    h = totalseconds // 3600
    m = (totalseconds - h*3600) // 60
    s = totalseconds - h*3600 - m*60
    print('eval end in time {}h {}m {:.4f}s'.format(h,m,s))
    # print(InterpretAll.shape,Y.shape)


    try:
        os.makedirs(f'../roc_cruve_data/figs/{opt.detect_method}')
    except:
        pass
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:size]<threshold).sum() + (Z[size:]>=threshold).sum()
    acc = acc/2/size
    print('auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(roc_auc,threshold,acc))
    save_roc_cruve(Y,Z,f'../roc_cruve_data/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/{opt.detect_method}/Normal')
    save_roc_fig(Y,Z,f'../roc_cruve_data/figs/{opt.detect_method}/{opt.data_phase}_{opt.attack}_{opt.interpret_method}_{opt.detect_method}_Normal.png',
    title = f'{opt.data_phase}_{opt.attack}_{opt.interpret_method}_{opt.detect_method}_AUC')

    Y = torch.cat([torch.ones(size),torch.zeros(size)],dim=0).numpy()  # 反向操作
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:size]>=threshold).sum() + (Z[size:]<threshold).sum()
    acc = acc/2/size
    print('Reverse :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(roc_auc,threshold,acc))
    save_roc_cruve(Y,Z,f'../roc_cruve_data/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/{opt.detect_method}/Reverse')
    save_roc_fig(Y,Z,f'../roc_cruve_data/figs/{opt.detect_method}/{opt.data_phase}_{opt.attack}_{opt.interpret_method}_{opt.detect_method}_Reverse.png',
    title = f'{opt.data_phase}_{opt.attack}_{opt.interpret_method}_{opt.detect_method}_AUC')


def auc_curve(y,prob,plt = True):
    fpr,tpr,thresholds = roc_curve(y,prob) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
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
    CleanDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/AdvDataSet.npy')
    Labels = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/Labels.npy')
    CleanInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/CleanInterpret.npy')
    AdvInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/AdvInterpret.npy')
    size = CleanInterpret.shape[0]

    for v in ['CleanDataSet','AdvDataSet','CleanInterpret','AdvInterpret']:
        n = eval(v)
        try:
            os.makedirs(f'../maps/{opt.interpret_method}/{opt.attack}')
        except Exception as e:
            print(f'../maps/{opt.interpret_method}/{opt.attack}  already exist')

        vutils.save_image(n[:64],f'../maps/{opt.interpret_method}/{opt.attack}/{v}.jpg',nrow=8)
    
    x = CleanDataSet-AdvDataSet
    x = x.view(size,-1)
    l2 = torch.norm(x,dim=1,p=2).mean().item()
    print('l2 distance is {:.4f}'.format(l2))

def redraw():
    Z = pk_load(f'../roc_cruve_data/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/{opt.detect_method}/Normal/prob')
    Y = pk_load(f'../roc_cruve_data/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/{opt.detect_method}/Normal/y')
    save_roc_fig(Y,Z,f'../roc_cruve_data/figs/{opt.detect_method}/{opt.data_phase}_{opt.attack}_{opt.interpret_method}_{opt.detect_method}_Normal.png',
        title = f'{opt.data_phase}_{opt.attack}_{opt.interpret_method}_{opt.detect_method}_AUC')

    Z = pk_load(f'../roc_cruve_data/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/{opt.detect_method}/Reverse/prob')
    Y = pk_load(f'../roc_cruve_data/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/{opt.detect_method}/Reverse/y')
    save_roc_fig(Y,Z,f'../roc_cruve_data/figs/{opt.detect_method}/{opt.data_phase}_{opt.attack}_{opt.interpret_method}_{opt.detect_method}_Reverse.png',
        title = f'{opt.data_phase}_{opt.attack}_{opt.interpret_method}_{opt.detect_method}_AUC')


if __name__ == '__main__':
    print("general_detect.py")
    opt = Options().parse_arguments()
    t1 = datetime.now()
    opt.dataset = 'cifar10'
    # opt.dataset = 'fmnist'
    opt.gpu_id = 0
    
    opt.weight_decay=5e-4
    # opt.GAUS=0.05 # fmnist -> 0.1
    # opt.loss2_alp=0.5
    opt.num_epoches = 40
    opt.lr = 0.007
    opt.lr_step = 10
    opt.lr_gamma = 0.8

    opt.pgd_eps = 0.015
    opt.pgd_eps_iter = 0.001
    opt.pgd_iterations = 200
    opt.fgsm_eps = 0.02
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
        opt.workers=16
    else:
        opt.train_batchsize = opt.val_batchsize = 170
        # opt.classifier_net = 'vgg11bn'
        opt.workers=4

    opt.classifier_net = 'vgg11bn'
    opt.device = torch.device("cuda:{}".format(opt.gpu_id))
    opt.interpret_method = 'LRP'
    opt.data_phase = 'test'
    opt.detect_method = 'iforest' # ['iforest','SVDD','LOF','Envelope'] # Envelope 太难跑
    opt.minus = False
    opt.scale = 1
    report_args(opt)

    # opt.attack='PGD-T'
    # for opt.pgd_eps in [0.013,0.015,0.02]:
    #     for opt.pgd_eps_iter in [0.003,0.004,0.005]:
    #         acc = generate(inter = False)
    #         print('^'*35,'pgd_eps',opt.pgd_eps,'pgd_eps_iter',opt.pgd_eps_iter,'acc={:.4f}'.format(acc))
    
    opt.attack='FGSM-U'
    for opt.fgsm_eps in [0.2,0.15,0.1,0.08,0.06,0.04,0.03,0.02,0.01,0.007]:
        acc = generate(inter=False)
        print('^'*35,'fgsm_eps',opt.fgsm_eps,'acc={:.4f}'.format(acc))
        
    # test(opt)
    # generate()
    # for opt.attack in ['CW-U','PGD-U','ADV2-T']:
    # for opt.attack in ['FGSM-U', 'PGD-U', 'PGD-T', 'DFool-U', 'CW-U', 'CW-T', 'DDN-U', 'DDN-T']:
    # for opt.detect_method in ['iforest']:
    #     for opt.attack in [ 'PGD-T','PGD-U','FGSM-U',]:
    #         generate()
            # for opt.interpret_method in ['LRP','VG','IG','GBP']:
            # #     print("+"*10)
            #     print(f"attack is {opt.attack}, interpret is {opt.interpret_method}, detect is {opt.detect_method}, scale is {opt.scale}")
            #     generate_interpret_only()
            #     make_graph()
            #     redraw()
    # opt.attack = 'CW-U'
    # main(train=False)
    # opt.interpret_method = 'VG'
    # for opt.attack in ['ADV2-T','PGD-U','CW-U']:
    #     main(train=True)
    