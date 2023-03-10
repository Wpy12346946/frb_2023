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

    CleanDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
        
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
        os.makedirs(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}')
    except Exception as e:
        print(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}  already exist')

    try:
        os.makedirs(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}')
    except Exception as e:
        print(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}  already exist')

    for v in ['CleanInterpret','AdvInterpret']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/{v}.npy')

    return

def generate_result_only():
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

    CleanDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
        
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
        os.makedirs(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}')
    except Exception as e:
        print(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}  already exist')

    for v in ['CleanResult','AdvResult']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')
    return

def generate():
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

        interpreter = interpretermethod(model, opt.interpret_method)
        saliency_images = interpreter.interpret(data)
        interpreter.release()
        CleanInterpret.append(saliency_images.clone().cpu())

        interpreter = interpretermethod(model, opt.interpret_method)
        saliency_images = interpreter.interpret(adv_data)
        interpreter.release()
        AdvInterpret.append(saliency_images.clone().cpu())
        
    print("adv_correct = {:.4f} , org_correct = {:.4f}".format(correct.item()/size,org_correct.item()/size))

    try:
        os.makedirs(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}')
    except Exception as e:
        print(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}  already exist')

    try:
        os.makedirs(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}')
    except Exception as e:
        print(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}  already exist')
        
    for v in ['CleanDataSet','AdvDataSet','Labels']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')

    for v in ['CleanInterpret','AdvInterpret']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/{v}.npy')



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



class PCA_Reducer:
    def __init__(self):
        self.eigen_vecs = None

    def train(self,X):
        # Data matrix X, assumes 0-centered
        n, m = X.shape
        X = X - X.mean(axis=0)
        # Compute covariance matrix
        C = np.dot(X.T, X) / (n-1)
        # Eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eig(C)
        # Project X onto PC space
        self.eigen_vecs = eigen_vecs
        X_pca = np.dot(X, eigen_vecs)
        return X_pca

    def reduce(self,X):
        return np.dot(X,self.eigen_vecs)

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
    generate_result_only()
    CleanDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    if opt.interpret_method=='org':
        CleanInterpret = CleanDataSet
        AdvInterpret = AdvDataSet
    else:
        CleanInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
        AdvInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')
    CleanResult = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanResult.npy')
    AdvResult = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvResult.npy')

    CleanDataSet = CleanDataSet[CleanResult==Labels]
    AdvDataSet = AdvDataSet[(CleanResult==Labels)&(AdvResult!=Labels)]
    CleanInterpret = CleanInterpret[CleanResult==Labels]
    AdvInterpret = AdvInterpret[(CleanResult==Labels)&(AdvResult!=Labels)]
    return CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels

def get_difference(p):
    CleanDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    x = CleanDataSet-AdvDataSet
    x = x.view(x.shape[0],-1)
    l = torch.norm(x,dim=1,p=p).mean().item()
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

class Ensembler:
    def __init__(self,interpret_methods,detect_method='iforest'):
        self.interpret_methods = interpret_methods
        self.detect_method = detect_method
        self.clf_list = []
        self.result = {}
        rng = np.random.RandomState(420)
        for i_method in interpret_methods:
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
            self.clf_list.append(clf)
    
    def save(self,dir_path):
        make_dir(dir_path)
        for i_method,clf in zip(self.interpret_methods,self.clf_list):
            pk_dump(clf,os.path.join(dir_path,f'{i_method}.pth'))
        
    def load(self,dir_path,use_normers = False):
        for ind,i_method in enumerate(self.interpret_methods):
            clf = pk_load(os.path.join(dir_path,f'{i_method}.pth'))
            self.clf_list[ind] = clf
        if use_normers:
            normers = {}
            for opt.interpret_method in self.interpret_methods:
                normalizer = Normalizer()
                normalizer.load(f'../classifier_pth/iforest_clf-ensemble/normer/{opt.interpret_method}')
                normers[opt.interpret_method] = normalizer
            self.normers = normers

    def fit(self,interpret_method,X_train):
        for i_method,clf in zip(self.interpret_methods,self.clf_list):
            if i_method == interpret_method:
                clf.fit(X_train)
                break
    
    def calculate(self,interpret_method,X,use_normers=False):
        for i_method,clf in zip(self.interpret_methods,self.clf_list):
            if i_method == interpret_method:
                Z = clf.decision_function(X)
                if use_normers and self.normers is not None:
                    Z = self.normers[i_method].get(Z)
                self.result[i_method] = Z
                break

    def ensemble(self,ensemble_method):
        for k,v in self.result.items():
            length = len(v)
            break

        if ensemble_method == 'Sum-Up':
            Z = 0
            for k,v in self.result.items():
                Z += v
        elif(ensemble_method == 'Breadth-First'):
            sorted_result = []
            sorted_index = []
            for k,v in self.result.items():
                sorted_result.append(np.sort(v)) # 升序，靠前接近黑
                sorted_index.append(np.argsort(v))
                # sorted_result.append(np.sort(v)[::-1]) # 降序，靠前接近白
                # sorted_index.append(np.argsort(v)[::-1])
            S = []
            S_ind = []
            for ind in range(length):
                for r,i in zip(sorted_result,sorted_index):
                    if i[ind] not in S_ind:
                        S_ind.append(i[ind])
                        S.append(r[ind])
            Z = np.zeros_like(sorted_result[0])
            for idx,i in enumerate(S_ind):
                Z[i] = S[idx]
        elif(ensemble_method == 'Max'):
            Z = []
            for ind in range(length):
                max = -1e10
                for k,r in self.result.items():
                    if r[ind]>max:
                        max = r[ind]
                Z.append(max)
            Z = np.array(Z)
        elif(ensemble_method == 'Min'):
            Z = []
            for ind in range(length):
                min = 1e10
                for k,r in self.result.items():
                    if r[ind]<min:
                        min = r[ind]
                Z.append(min)
            Z = np.array(Z)
        else:
            raise Exception(f"{self.ensemble_method} not implemented")
        return Z


from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import sklearn
def main(interpret_methods=['DL','LRP','org','VG','GBP','IG'],train = True):
    reducer = PCA_Reducer()
    ensembled_clf = Ensembler(interpret_methods=interpret_methods,detect_method='iforest')
    for opt.interpret_method in interpret_methods:
        generate_interpret_only()
        CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
        cleansize = CleanInterpret.shape[0]
        advsize = AdvInterpret.shape[0]
        total = Labels.shape[0]
        print(f"i_method is {opt.interpret_method}, total is {total}, clean size is {cleansize} (clean/total={cleansize/total}), adv size is {advsize} (adv/clean={advsize/cleansize})")

        clf_path_name = f'../classifier_pth/{opt.detect_method}_clf-ensemble'
        if train:
            print('training begin')
            t1 = datetime.now()
            X_train = CleanInterpret.view(cleansize,-1).numpy()
            X_train = reducer.train(X_train)
            ensembled_clf.fit(opt.interpret_method,X_train)
            t2 = datetime.now()
            totalseconds = (t2-t1).total_seconds()
            h = totalseconds // 3600
            m = (totalseconds - h*3600) // 60
            s = totalseconds - h*3600 - m*60
            print('training end in time {}h {}m {:.4f}s'.format(h,m,s))
            ensembled_clf.save(clf_path_name)
        else:
            X_train = CleanInterpret.view(cleansize,-1).numpy()
            X_train = reducer.train(X_train)
            ensembled_clf.load(clf_path_name)

        print('eval begin')
        t1 = datetime.now()
        InterpretAll = torch.cat([CleanInterpret,AdvInterpret],dim=0).view(cleansize+advsize,-1).numpy()
        InterpretAll = reducer.reduce(InterpretAll)
        ensembled_clf.calculate(opt.interpret_method,InterpretAll)
        t2 = datetime.now()
        totalseconds = (t2-t1).total_seconds()
        h = totalseconds // 3600
        m = (totalseconds - h*3600) // 60
        s = totalseconds - h*3600 - m*60
        print('eval end in time {}h {}m {:.4f}s'.format(h,m,s))

    Z = ensembled_clf.ensemble('Breadth-First')

    Y = torch.cat([torch.zeros(cleansize),torch.ones(advsize)],dim=0).numpy()
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]<threshold).sum() + (Z[cleansize:]>=threshold).sum()
    acc = acc/(cleansize+advsize)
    print('Reverse :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(roc_auc,threshold,acc))
    roc_auc_rev= roc_auc

    Y = torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy()  # 反向操作
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]>=threshold).sum() + (Z[cleansize:]<threshold).sum()
    acc = acc/(cleansize+advsize)
    print('Normal :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(roc_auc,threshold,acc))

    if roc_auc>roc_auc_rev:
        auc_breadth = roc_auc
    else:
        auc_breadth = roc_auc_rev
    

    Z = ensembled_clf.ensemble('Sum-Up')

    Y = torch.cat([torch.zeros(cleansize),torch.ones(advsize)],dim=0).numpy()
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]<threshold).sum() + (Z[cleansize:]>=threshold).sum()
    acc = acc/(cleansize+advsize)
    print('Reverse :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(roc_auc,threshold,acc))
    roc_auc_rev= roc_auc

    Y = torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy()  # 反向操作
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]>=threshold).sum() + (Z[cleansize:]<threshold).sum()
    acc = acc/(cleansize+advsize)
    print('Normal :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(roc_auc,threshold,acc))

    if roc_auc>roc_auc_rev:
        auc_sum = roc_auc
    else:
        auc_sum = roc_auc_rev

    Z = ensembled_clf.ensemble('Min')

    Y = torch.cat([torch.zeros(cleansize),torch.ones(advsize)],dim=0).numpy()
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]<threshold).sum() + (Z[cleansize:]>=threshold).sum()
    acc = acc/(cleansize+advsize)
    print('Reverse :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(roc_auc,threshold,acc))
    roc_auc_rev= roc_auc

    Y = torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy()  # 反向操作
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]>=threshold).sum() + (Z[cleansize:]<threshold).sum()
    acc = acc/(cleansize+advsize)
    print('Normal :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(roc_auc,threshold,acc))

    if roc_auc>roc_auc_rev:
        auc_min = roc_auc
    else:
        auc_min = roc_auc_rev

    Z = ensembled_clf.ensemble('Max')

    Y = torch.cat([torch.zeros(cleansize),torch.ones(advsize)],dim=0).numpy()
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]<threshold).sum() + (Z[cleansize:]>=threshold).sum()
    acc = acc/(cleansize+advsize)
    print('Reverse :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(roc_auc,threshold,acc))
    roc_auc_rev= roc_auc

    Y = torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy()  # 反向操作
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]>=threshold).sum() + (Z[cleansize:]<threshold).sum()
    acc = acc/(cleansize+advsize)
    print('Normal :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(roc_auc,threshold,acc))

    if roc_auc>roc_auc_rev:
        auc_max = roc_auc
    else:
        auc_max = roc_auc_rev


    print("auc_sum is {:.4f}, auc_breadth is {:.4f}, auc_min is {:.4f}, auc_max is {:.4f}".format(auc_sum,auc_breadth,auc_min,auc_max))
    return auc_sum,auc_breadth,auc_min,auc_max

def main_once(train = True):
    # CleanDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/CleanDataSet.npy')
    # AdvDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/AdvDataSet.npy')
    # Labels = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/Labels.npy')
    # if opt.interpret_method=='org':
    #     CleanInterpret = CleanDataSet
    #     AdvInterpret = AdvDataSet
    # else:
    #     CleanInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/CleanInterpret.npy')
    #     AdvInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}/{opt.interpret_method}/AdvInterpret.npy')
    CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
    cleansize = CleanInterpret.shape[0]
    advsize = AdvInterpret.shape[0]
    total = Labels.shape[0]
    print(f"total is {total}, clean size is {cleansize} (clean/total={cleansize/total}), adv size is {advsize} (adv/clean={advsize/cleansize})")

    # for v in ['CleanDataSet','AdvDataSet','CleanInterpret','AdvInterpret','Labels']:
    #     n = eval(v)
    #     print(f"{v} shape is : {n.shape}")

    clf_path_name = f'../classifier_pth/{opt.detect_method}_clf-{opt.interpret_method}.pkl'
    if train:
        print('training begin')
        t1 = datetime.now()
        rng = np.random.RandomState(420)
        X_train = CleanInterpret.view(cleansize,-1).numpy()
        reducer = PCA_Reducer()
        X_train = reducer.train(X_train)
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
        X_train = CleanInterpret.view(cleansize,-1).numpy()
        reducer = PCA_Reducer()
        X_train = reducer.train(X_train)

    print('eval begin')
    t1 = datetime.now()
    InterpretAll = torch.cat([CleanInterpret,AdvInterpret],dim=0).view(cleansize+advsize,-1).numpy()
    InterpretAll = reducer.reduce(InterpretAll)
    Z = clf.decision_function(InterpretAll)
    Y = torch.cat([torch.zeros(cleansize),torch.ones(advsize)],dim=0).numpy()
    # Y = torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy()  # 反向操作
    Y = Y
    t2 = datetime.now()
    totalseconds = (t2-t1).total_seconds()
    h = totalseconds // 3600
    m = (totalseconds - h*3600) // 60
    s = totalseconds - h*3600 - m*60
    print('eval end in time {}h {}m {:.4f}s'.format(h,m,s))
    # print(InterpretAll.shape,Y.shape)

    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]<threshold).sum() + (Z[cleansize:]>=threshold).sum()
    acc = acc/(cleansize+advsize)
    print('Reverse :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(roc_auc,threshold,acc))
    save_roc_cruve(Y,Z,f'../roc_cruve_data/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/PCA-{opt.detect_method}/Reverse')
    save_roc_fig(Y,Z,f'../roc_cruve_data/figs/{opt.data_phase}_{opt.attack}_{opt.attack_param}_{opt.interpret_method}_PCA-{opt.detect_method}_Reverse.png')
    roc_auc_rev= roc_auc

    Y = torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy()  # 反向操作
    roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
    acc = (Z[:cleansize]>=threshold).sum() + (Z[cleansize:]<threshold).sum()
    acc = acc/(cleansize+advsize)
    print('Normal :: auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(roc_auc,threshold,acc))
    save_roc_cruve(Y,Z,f'../roc_cruve_data/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/PCA-{opt.detect_method}/Normal')
    save_roc_fig(Y,Z,f'../roc_cruve_data/figs/{opt.data_phase}_{opt.attack}_{opt.attack_param}_{opt.interpret_method}_PCA-{opt.detect_method}_Normal.png')

    if roc_auc>roc_auc_rev:
        return roc_auc
    else:
        return roc_auc_rev


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
    CleanDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    CleanInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
    AdvInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')
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

if __name__ == '__main__':
    print("pca_detect.py")
    opt = Options().parse_arguments()
    t1 = datetime.now()
    opt.dataset = 'cifar10'
    # opt.dataset = 'fmnist'
    opt.gpu_id = 1
    opt.device = torch.device("cuda:{}".format(opt.gpu_id))
    
    opt.weight_decay=5e-4
    # opt.GAUS=0.05 # fmnist -> 0.1
    # opt.loss2_alp=0.5
    opt.num_epoches = 40
    opt.lr = 0.007
    opt.lr_step = 10
    opt.lr_gamma = 0.8

    opt.pgd_eps = 0.008
    opt.pgd_eps_iter = 0.0002
    opt.pgd_iterations = 1000
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
        opt.workers=16
    else:
        opt.train_batchsize = opt.val_batchsize = 128
        # opt.classifier_net = 'vgg11bn'
        opt.workers=4

    opt.classifier_net = 'vgg11bn'
    # opt.device = torch.device("cuda:{}".format(opt.gpu_id))
    opt.interpret_method = 'LRP'
    opt.data_phase = 'test'
    opt.detect_method = 'iforest' # ['iforest','SVDD','LOF','Envelope'] # Envelope 太难跑
    opt.minus = False
    opt.scale = 1
    report_args(opt)


    opt.attack = 'CW-U'
    opt.cw_confidence = 9

    # opt.attack = 'PGD-U'
    # for opt.pgd_eps in [0.005]:
    #     opt.attack_param = opt.pgd_eps
    #     opt.interpret_method = 'org'
    #     generate_interpret_only()
    #     auc = main_once(train=True)
    #     print(auc)

    # for opt.attack in [ 'PGD-U']:
    #     auc_values = {}
    #     for opt.pgd_eps in [0.005,0.007,0.009,0.01,0.013,0.015]:
    #         opt.attack_param = opt.pgd_eps
    #         CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
    #         attack_rate = AdvInterpret.shape[0]/CleanInterpret.shape[0]
    #         item = {'attack_rate':attack_rate}
    #         item['l2'] = get_difference(2)
    #         for opt.interpret_method in ['DL','LRP','org','VG','GBP','IG']:
    #             print("+"*10)
    #             generate_interpret_only()
    #             auc = main_once(train=True)
    #             item[opt.interpret_method] = auc
    #         auc_sum, auc_breadth,auc_min,auc_max = main(train=True,interpret_methods = ['DL','LRP','org','VG','GBP','IG'])
    #         item['ensemble_sum_up'] = auc_sum
    #         item['ensemble_breadth_first'] = auc_breadth
    #         item['ensemble_min'] = auc_min
    #         item['ensemble_max'] = auc_max
    #         auc_values[opt.pgd_eps] = item
    #     print(auc_values)
    #     pk_dump(auc_values,f'../tmp_dataset/attack_eps/{opt.attack}.pth')

    # auc_values = pk_load(f'../tmp_dataset/attack_eps/PGD-U.pth')
    # plot_auc_eps(auc_values,f'../tmp_dataset/attack_eps/fig-pgd-u.jpg',methods=['DL','LRP','org','VG','GBP','IG', 'ensemble_sum_up','ensemble_breadth_first','ensemble_min','ensemble_max','attack_rate'])
    # auc_values = pd.DataFrame.from_dict(auc_values)
    # auc_values.to_csv(f'../tmp_dataset/attack_eps/PGD-U.csv',float_format = '%.3f')
    # print(auc_values)

    auc_values = {}
    

    for opt.fgsm_eps in [0.04,0.035,0.03,0.025,0.02,0.015,0.01,0.008,0.005,0.001]:
        opt.attack = 'FGSM-U'
        opt.param = opt.attack_param = opt.fgsm_eps
        # generate()
        CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
        attack_rate = AdvInterpret.shape[0]/CleanInterpret.shape[0]
        item = {'attack_rate':attack_rate}
        item['l2'] = get_difference(2)
        auc_sum, auc_breadth,auc_min,auc_max = main(train=False,interpret_methods = ['DL','LRP','org','VG','GBP','IG'])
        item['ensemble_sum_up'] = auc_sum
        # item['ensemble_breadth_first'] = auc_breadth
        item['ensemble_min'] = auc_min
        item['ensemble_max'] = auc_max
        for opt.interpret_method in ['DL','LRP','org','VG','GBP','IG']:
            generate_interpret_only()
            auc = main_once(train=False)
            item[opt.interpret_method] = auc
        auc_values[opt.fgsm_eps] = item
    print(auc_values)
    make_dir('../tmp_dataset/attack_eps/')
    pk_dump(auc_values,f'../tmp_dataset/attack_eps/{opt.attack}.pth')

    auc_values = pk_load(f'../tmp_dataset/attack_eps/FGSM-U.pth')
    plot_auc_eps(auc_values,f'../tmp_dataset/attack_eps/fig-fgsm-u.jpg',methods=['DL','LRP','org','VG','GBP','IG', 'ensemble_sum_up','ensemble_min','ensemble_max','attack_rate'])
    auc_values = pd.DataFrame.from_dict(auc_values)
    auc_values.to_csv(f'../tmp_dataset/attack_eps/FGSM-U.csv',float_format = '%.3f')
    print(auc_values)
    exit()


    #pgd
    # for opt.attack in [ 'PGD-T','PGD-U']:
    #     auc_values = {}
    #     for opt.pgd_eps in [0.005,0.007,0.009,0.01,0.013,0.015]:
    #         generate()
    #         CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
    #         attack_rate = AdvInterpret.shape[0]/CleanInterpret.shape[0]
    #         item = {'attack_rate':attack_rate}
    #         item['l2'] = get_difference(2)
    #         auc_sum, auc_breadth,auc_min,auc_max = main(train=True,interpret_methods = ['DL','LRP','org','VG','GBP','IG'])
    #         item['ensemble_sum_up'] = auc_sum
    #         item['ensemble_breadth_first'] = auc_breadth
    #         item['ensemble_min'] = auc_min
    #         item['ensemble_max'] = auc_max
    #         for opt.interpret_method in ['DL','LRP','org','VG','GBP','IG']:
    #             print("+"*10)
    #             generate_interpret_only()
    #             auc = main_once(train=True)
    #             item[opt.interpret_method] = auc
    #         auc_values[opt.pgd_eps] = item
    #     print(auc_values)
    #     pk_dump(auc_values,f'../tmp_dataset/attack_eps/{opt.attack}.pth')

    auc_values = pk_load(f'../tmp_dataset/attack_eps/PGD-T.pth')
    plot_auc_eps(auc_values,f'../tmp_dataset/attack_eps/fig-pgd-t.jpg',methods=['DL','LRP','org','VG','GBP','IG', 'ensemble_sum_up','ensemble_breadth_first','ensemble_min','ensemble_max','attack_rate'])
    auc_values = pd.DataFrame.from_dict(auc_values)
    auc_values.to_csv(f'../tmp_dataset/attack_eps/PGD-T.csv',float_format = '%.3f')
    print(auc_values)

    auc_values = pk_load(f'../tmp_dataset/attack_eps/PGD-U.pth')
    plot_auc_eps(auc_values,f'../tmp_dataset/attack_eps/fig-pgd-u.jpg',methods=['DL','LRP','org','VG','GBP','IG', 'ensemble_sum_up','ensemble_breadth_first','ensemble_min','ensemble_max','attack_rate'])
    auc_values = pd.DataFrame.from_dict(auc_values)
    auc_values.to_csv(f'../tmp_dataset/attack_eps/PGD-U.csv',float_format = '%.3f')
    print(auc_values)

    # cw
    for opt.attack in [ 'CW-U','CW-T']:
        # auc_values = {}
        # for opt.cw_confidence in [0.1,0.3,0.5,1,3,6,9,15]:
        #     generate()
        #     CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
        #     attack_rate = AdvInterpret.shape[0]/CleanInterpret.shape[0]
        #     item = {'attack_rate':attack_rate}
        #     l2 = get_difference(2)
        #     item['l2'] = l2
        #     auc_sum, auc_breadth,auc_min,auc_max = main(train=True,interpret_methods = ['DL','LRP','org','VG','GBP','IG'])
        #     item['ensemble_sum_up'] = auc_sum
        #     item['ensemble_breadth_first'] = auc_breadth
        #     item['ensemble_min'] = auc_min
        #     item['ensemble_max'] = auc_max
        #     for opt.interpret_method in ['DL','LRP','org','VG','GBP','IG']:
        #         print("+"*10)
        #         generate_interpret_only()
        #         auc = main_once(train=True)
        #         item[opt.interpret_method] = auc
        #         print('attack is {}, param is {:.4f}, attack rate is {:.4f}, l2 is {:.4f}, interpret is {}, auc is {:.4f}'.format(
        #             opt.attack,opt.cw_confidence,attack_rate,l2,opt.interpret_method,auc
        #         ))
        #     auc_values[opt.cw_confidence] = item
        # print(auc_values)
        # pk_dump(auc_values,f'../tmp_dataset/attack_eps/{opt.attack}.pth')
        auc_values = pk_load(f'../tmp_dataset/attack_eps/{opt.attack}.pth')
        plot_auc_eps(auc_values,f'../tmp_dataset/attack_eps/fig-{opt.attack}.jpg',methods=['DL','LRP','org','VG','GBP','IG', 'ensemble_sum_up','ensemble_breadth_first','ensemble_min','ensemble_max','attack_rate'])
        auc_values = pd.DataFrame.from_dict(auc_values)
        auc_values.to_csv(f'../tmp_dataset/attack_eps/{opt.attack}.csv',float_format = '%.3f')



    # opt.pgd_eps_iter = 0.0007
    # opt.pgd_iterations = 400
    # for opt.attack in [ 'ADV2-T']:
    #     auc_values = {}
    #     for opt.pgd_eps in [0.001,0.003,0.005,0.007,0.009,0.01,0.013,0.015]:
    #         generate()
    #         CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
    #         attack_rate = AdvInterpret.shape[0]/CleanInterpret.shape[0]
    #         item = {'attack_rate':attack_rate}
    #         l2 = get_difference(2)
    #         item['l2'] = l2
    #         for opt.interpret_method in ['DL','LRP','org','VG','GBP','IG']:
    #             print("+"*10)
    #             generate_interpret_only()
    #             auc = main(train=True)
    #             item[opt.interpret_method] = auc
    #             print('attack is {}, param is {:.4f}, attack rate is {:.4f}, l2 is {:.4f}, interpret is {}, auc is {:.4f}'.format(
    #                 opt.attack,opt.pgd_eps,attack_rate,l2,opt.interpret_method,auc
    #             ))
    #         auc_values[opt.pgd_eps] = item
    #     print(auc_values)
    #     pk_dump(auc_values,f'../tmp_dataset/attack_eps/{opt.attack}.pth')

    #     plot_auc_eps(auc_values,f'../tmp_dataset/attack_eps/{opt.attack}.jpg',methods = ['DL','LRP','org','VG','GBP','IG','attack_rate'])
    #     auc_values = pd.DataFrame.from_dict(auc_values)
    #     auc_values.to_csv(f'../tmp_dataset/attack_eps/{opt.attack}.csv',float_format = '%.3f')


    opt.pgd_eps_iter = 0.0007
    opt.pgd_iterations = 400
    for opt.attack in ['ADV2-T']:
        auc_values = {}
        for opt.pgd_eps in [0.001,0.003,0.005,0.007,0.009,0.01,0.013,0.015]:
            generate()
            CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
            attack_rate = AdvInterpret.shape[0]/CleanInterpret.shape[0]
            item = {'attack_rate':attack_rate}
            item['l2'] = get_difference(2)
            auc_sum, auc_breadth,auc_min,auc_max = main(train=True,interpret_methods = ['DL','LRP','org','VG','GBP','IG'])
            item['ensemble_sum_up'] = auc_sum
            item['ensemble_breadth_first'] = auc_breadth
            item['ensemble_min'] = auc_min
            item['ensemble_max'] = auc_max
            for opt.interpret_method in ['DL','LRP','org','VG','GBP','IG']:
                print("+"*10)
                generate_interpret_only()
                auc = main_once(train=True)
                item[opt.interpret_method] = auc
            auc_values[opt.pgd_eps] = item
        print(auc_values)
        pk_dump(auc_values,f'../tmp_dataset/attack_eps/{opt.attack}.pth')

        auc_values = pk_load(f'../tmp_dataset/attack_eps/{opt.attack}.pth')
        plot_auc_eps(auc_values,f'../tmp_dataset/attack_eps/fig-{opt.attack}.jpg',methods=['DL','LRP','org','VG','GBP','IG', 'ensemble_sum_up','ensemble_breadth_first','ensemble_min','ensemble_max','attack_rate'])
        auc_values = pd.DataFrame.from_dict(auc_values)
        auc_values.to_csv(f'../tmp_dataset/attack_eps/{opt.attack}.csv',float_format = '%.3f')

