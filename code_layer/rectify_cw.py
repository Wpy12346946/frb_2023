import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from Options import Options,report_args
import os
from datasets import *
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
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
from PIL import Image
from interpreter_methods import interpretermethod
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import sklearn
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

@torch.enable_grad()
def getRec(classifier,data, label):
    data=data.detach().clone()
    data.requires_grad=True
    clean_pred = classifier(data).max(1)[1]
    attack = opt.attack
    cw_max_iterations = opt.cw_max_iterations
    cw_confidence = opt.cw_confidence
    opt.attack = 'CW-T'
    opt.cw_max_iterations=10
    opt.cw_confidence=9
    attack_label = label

    perturbed_data = adversarialattack(opt.attack, classifier, data, attack_label, opt)
    opt.attack = attack
    opt.cw_max_iterations = cw_max_iterations
    opt.cw_confidence = cw_confidence
    return perturbed_data


def make_dir(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(f'{path}  already exist')

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

    make_dir(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}')

    for v in ['CleanInterpret','AdvInterpret']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/{v}.npy')
    return

def generate_interpret_only_rec(lb):
    if opt.interpret_method == 'org':
        return
    # Initialize the network
    # choose classifier_net in package:models
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet-Rec-{lb}.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet-Rec-{lb}.npy')

    CleanInterpret = []
    AdvInterpret = []
    
    for d_name,i_name in [('AdvDataSet','AdvInterpret'),('CleanDataSet','CleanInterpret')]:
        dataset = eval(d_name)
        dataset = TensorDataset(dataset)
        dataloader = DataLoader(dataset = dataset, batch_size=opt.val_batchsize,shuffle=False)

        for data, in tqdm(dataloader, desc=d_name):
            # Send the data and label to the device
            data = data.to(opt.device)
            batch_size = data.shape[0]

            # Forward pass the data through the model
            output = model(data)
            interpreter = interpretermethod(model, opt.interpret_method)
            saliency_images = interpreter.interpret(data)
            interpreter.release()
            eval(i_name).append(saliency_images.clone().cpu())

    make_dir(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}')

    for v in ['CleanInterpret','AdvInterpret']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/{v}-Rec-{lb}.npy')

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

    for v in ['CleanResult','AdvResult']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')
    return

def generate():
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
        
    print("adv_correct = {:.4f} , org_correct = {:.4f}".format(correct.item()/size,org_correct.item()/size))

    make_dir(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}')

        
    for v in ['CleanDataSet','AdvDataSet','Labels']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')

    return

def get_sift_data():
    generate_result_only()
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
    CleanInterpret = CleanInterpret[CleanResult==Labels]
    AdvInterpret = AdvInterpret[(CleanResult==Labels)&(AdvResult!=Labels)]
    return CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels

def get_difference(p):
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    x = CleanDataSet-AdvDataSet
    x = x.view(x.shape[0],-1)
    l = torch.norm(x,dim=1,p=p).mean().item()
    return l

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
    
    def dump(self,path):
        pk_dump(self.eigen_vecs,path)

    def load(self,path):
        self.eigen_vecs = pk_load(path)

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
        
    def load(self,dir_path):
        for ind,i_method in enumerate(self.interpret_methods):
            clf = pk_load(os.path.join(dir_path,f'{i_method}.pth'))
            self.clf_list[ind] = clf

    def fit(self,interpret_method,X_train):
        for i_method,clf in zip(self.interpret_methods,self.clf_list):
            if i_method == interpret_method:
                clf.fit(X_train)
                return clf
    
    def find_clf(self,interpret_method):
        for i_method,clf in zip(self.interpret_methods,self.clf_list):
            if i_method == interpret_method:
                return clf
    
    def calculate(self,interpret_method,X):
        for i_method,clf in zip(self.interpret_methods,self.clf_list):
            if i_method == interpret_method:
                Z = clf.decision_function(X)
                self.result[i_method] = Z
                break

    def calculate_once_min(self,Xmap):
        result = 1e100
        result_method = None
        for i_method,clf in zip(self.interpret_methods,self.clf_list):
            Z = clf.decision_function(Xmap[i_method])
            if result>Z:
                result = Z 
                result_method = i_method
        return result,result_method

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


def get_reducer_and_ensemble_clf(train=True,interpret_methods=['DL','LRP','VG','GBP','IG','org']):
    reducers = {}
    ensembled_clf = Ensembler(interpret_methods=interpret_methods,detect_method='iforest')
    if not train:
        ensembled_clf.load('../classifier_pth/iforest_clf-ensemble/') 
        for opt.interpret_method in interpret_methods:
            reducer = PCA_Reducer()
            reducer.load(f'../classifier_pth/iforest_clf-ensemble/reducers/{opt.interpret_method}.pth')
            reducers[opt.interpret_method] = reducer 
        return ensembled_clf,reducers
    
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    CleanResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanResult.npy')
    AdvResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvResult.npy')
    for opt.interpret_method in interpret_methods:
        print('train',opt.interpret_method)
        if opt.interpret_method=='org':
            CleanInterpret = CleanDataSet.clone()
            AdvInterpret = AdvDataSet.clone()
        else:
            # generate_interpret_only()
            try:
                CleanInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
                AdvInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')
            except:
                generate_interpret_only()
                CleanInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
                AdvInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')
        CleanInterpret = CleanInterpret[CleanResult==Labels]
        AdvInterpret = AdvInterpret[(CleanResult==Labels)&(AdvResult!=Labels)]

        X_train = CleanInterpret.clone().cpu().view(CleanInterpret.shape[0],-1).numpy()
        reducer = PCA_Reducer()
        X_train = reducer.train(X_train)
        reducers[opt.interpret_method] = reducer
        reducer.dump(f'../classifier_pth/iforest_clf-ensemble/reducers/{opt.interpret_method}.pth')
        clf = ensembled_clf.fit(opt.interpret_method,X_train)

        X_normal = CleanInterpret.view(CleanInterpret.shape[0],-1).numpy()
        X_abnormal = AdvInterpret.view(AdvInterpret.shape[0],-1).numpy()
        X_normal = reducer.reduce(X_normal)
        X_abnormal = reducer.reduce(X_abnormal)
        X_all = np.concatenate([X_normal,X_abnormal])
        Z = clf.decision_function(X_all)
        Y = torch.cat([torch.ones(CleanInterpret.shape[0]),torch.zeros(AdvInterpret.shape[0])],dim=0).numpy()
        roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
        acc = (Z[:CleanInterpret.shape[0]]>=threshold).sum() + (Z[CleanInterpret.shape[0]:]<threshold).sum()
        acc = acc/(CleanInterpret.shape[0]+AdvInterpret.shape[0])
        print('{}, auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(opt.interpret_method,roc_auc,threshold,acc))

    ensembled_clf.save('../classifier_pth/iforest_clf-ensemble/')
    return ensembled_clf,reducers

def generate_rectified(interpret_methods=['DL','LRP','VG','GBP','IG','org']):
    generate_result_only()
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    CleanResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanResult.npy')
    AdvResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvResult.npy')

    CleanDataSet = CleanDataSet[CleanResult==Labels]
    CleanLabels = Labels[CleanResult==Labels]
    AdvDataSet = AdvDataSet[(CleanResult==Labels)&(AdvResult!=Labels)]
    AdvLabels = Labels[(CleanResult==Labels)&(AdvResult!=Labels)]

    torch.save(CleanLabels,f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanLabels.npy')
    torch.save(AdvLabels,f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvLabels.npy')

    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    for lb in range(10):
        for dataset_name,labels_name in [('CleanDataSet','CleanLabels'),('AdvDataSet','AdvLabels')]:
            dataset = TensorDataset(eval(dataset_name),eval(labels_name))
            rec = []
            dataloader = DataLoader(dataset = dataset, batch_size=opt.val_batchsize,shuffle=False)
            for d,l in tqdm(dataloader,desc=f'{dataset_name} - rec - {lb}'):
                d = d.to(opt.device)
                tmp_label = l.clone()
                tmp_label[:]=lb
                tmp_label = tmp_label.to(opt.device)
                rec_img = getRec(model,d,tmp_label)
                rec.append(rec_img.clone().detach().cpu())
            rec = torch.cat(rec)
            print('{} rec shape is {}'.format(dataset_name,rec.shape))
            torch.save(rec,f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{dataset_name}-Rec-{lb}.npy')
            rec = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{dataset_name}-Rec-{lb}.npy')
            make_dir(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/sprate_{dataset_name}-Rec-{lb}')
            for ind in tqdm(range(rec.shape[0])):
                item = rec[ind].numpy()
                np.save(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/sprate_{dataset_name}-Rec-{lb}/{ind}.npy',item)

    for lb in range(10):
        for opt.interpret_method in interpret_methods:
            if opt.interpret_method == 'org':
                continue
            generate_interpret_only_rec(lb)
    #         CleanInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret-Rec-{lb}.npy')
    #         AdvInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret-Rec-{lb}.npy')
    #         print(CleanInterpret.shape[0],AdvInterpret.shape[0])
    #         make_dir(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/sprate_CleanInterpret-Rec-{lb}')
    #         for ind in tqdm(range(CleanInterpret.shape[0])):
    #             item = CleanInterpret[ind].numpy()
    #             np.save(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/sprate_CleanInterpret-Rec-{lb}/{ind}.npy',item)
    #         make_dir(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/sprate_AdvInterpret-Rec-{lb}')
    #         for ind in tqdm(range(AdvInterpret.shape[0])):
    #             item = AdvInterpret[ind].numpy()
    #             np.save(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/sprate_AdvInterpret-Rec-{lb}/{ind}.npy',item)


# 约登数寻找阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def auc_curve(y,prob,plt = True):
    fpr,tpr,thresholds = roc_curve(y,prob) ###计算真正率和假正率
    roc_auc = sklearn.metrics.auc(fpr,tpr) ###计算auc的值
    threshold,point = Find_Optimal_Cutoff(tpr,fpr,thresholds)

    return roc_auc,threshold,thresholds

def test_ensemble_reducer(interpret_methods=['DL','LRP','VG','GBP','IG','org']):
    ensembled_clf,reducers = get_reducer_and_ensemble_clf(train=False)
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    CleanResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanResult.npy')
    AdvResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvResult.npy')
    ths = {}

    for opt.interpret_method in interpret_methods:
        print('eval',opt.interpret_method)
        if opt.interpret_method=='org':
            CleanInterpret = CleanDataSet
            AdvInterpret = AdvDataSet
        else:
            CleanInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
            AdvInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')
        CleanInterpret = CleanInterpret[CleanResult==Labels]
        AdvInterpret = AdvInterpret[(CleanResult==Labels)&(AdvResult!=Labels)]
        X_normal = CleanInterpret.view(CleanInterpret.shape[0],-1).numpy()
        X_abnormal = AdvInterpret.view(AdvInterpret.shape[0],-1).numpy()

        X_normal = reducers[opt.interpret_method].reduce(X_normal)
        X_abnormal = reducers[opt.interpret_method].reduce(X_abnormal)
        X_all = np.concatenate([X_normal,X_abnormal])
        Z = ensembled_clf.find_clf(opt.interpret_method).decision_function(X_all)

        Y = torch.cat([torch.ones(CleanInterpret.shape[0]),torch.zeros(AdvInterpret.shape[0])],dim=0).numpy()
        roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
        acc = (Z[:CleanInterpret.shape[0]]>=threshold).sum() + (Z[CleanInterpret.shape[0]:]<threshold).sum()
        acc = acc/(CleanInterpret.shape[0]+AdvInterpret.shape[0])
        print('{}, auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(opt.interpret_method,roc_auc,threshold,acc))
        ths[opt.interpret_method] = threshold
    return ths

def get_prev_z(interpret_methods = ['DL','LRP','VG','GBP','IG','org']):
    ensembled_clf,reducers = get_reducer_and_ensemble_clf(train=False)
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    CleanResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanResult.npy')
    AdvResult = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvResult.npy')

    _clean = {}
    _adv = {}

    for i_method in interpret_methods:
        opt.interpret_method = i_method
        if os.path.exists(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret-Z.npy') and \
        os.path.exists(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret-Z.npy'):
            _clean[i_method] = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret-Z.npy')
            _adv[i_method] = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret-Z.npy')
            continue
        print(" get prev z begin",i_method)
        if opt.interpret_method == 'org':
            CleanInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
            AdvInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
        else:
            CleanInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
            AdvInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')

        CleanInterpret = CleanInterpret[CleanResult==Labels]
        AdvInterpret = AdvInterpret[(CleanResult==Labels)&(AdvResult!=Labels)]

        reducer = reducers[opt.interpret_method]
        clf = ensembled_clf.find_clf(opt.interpret_method)
        cleansize = CleanInterpret.shape[0]
        X_normal = CleanInterpret.view(cleansize,-1).numpy()
        X_normal = reducer.reduce(X_normal)
        Z_normal = clf.decision_function(X_normal)
        np.save(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret-Z.npy',Z_normal)

        advsize = AdvInterpret.shape[0]
        X_abnormal = AdvInterpret.view(advsize,-1).numpy()
        X_abnormal = reducer.reduce(X_abnormal)
        Z_abnormal = clf.decision_function(X_abnormal)
        np.save(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret-Z.npy',Z_abnormal)

        _clean[i_method] = Z_normal
        _adv[i_method] = Z_abnormal
    return _clean,_adv


def get_Z(interpret_methods=['DL','LRP','VG','GBP','IG','org']):
    ensembled_clf,reducers = get_reducer_and_ensemble_clf(train=False)
    for lb in range(10):
        for opt.interpret_method in interpret_methods:
            if opt.interpret_method == 'org':
                CleanInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet-Rec-{lb}.npy')
                AdvInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet-Rec-{lb}.npy')
            else:
                CleanInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret-Rec-{lb}.npy')
                AdvInterpret = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret-Rec-{lb}.npy')
            reducer = reducers[opt.interpret_method]
            clf = ensembled_clf.find_clf(opt.interpret_method)
            cleansize = CleanInterpret.shape[0]
            X_normal = CleanInterpret.view(cleansize,-1).numpy()
            X_normal = reducer.reduce(X_normal)
            Z_normal = clf.decision_function(X_normal)
            make_dir(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}')
            np.save(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret-Rec-Z-{lb}.npy',Z_normal)

            advsize = AdvInterpret.shape[0]
            X_abnormal = AdvInterpret.view(advsize,-1).numpy()
            X_abnormal = reducer.reduce(X_abnormal)
            Z_abnormal = clf.decision_function(X_abnormal)
            np.save(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret-Rec-Z-{lb}.npy',Z_abnormal)

            # Z = np.concatenate([Z_normal,Z_abnormal])
            # Y = torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy()
            # roc_auc,threshold,thresholds = auc_curve(Y,Z,plt=False)
            # acc = (Z_normal>=threshold).sum() + (Z_abnormal<threshold).sum()
            # acc = acc/(cleansize+advsize)
            # print('{}, auc is {:.4f}, threshold is {:.4f}, acc rate is {:.4f}'.format(opt.interpret_method,roc_auc,threshold,acc))

def test(interpret_methods = ['DL','LRP','VG','GBP','IG','org']):
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    Rec_clean={}
    Rec_adv = {}
    Z_adv = {}
    Z_clean = {}
    CleanLabels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanLabels.npy')
    AdvLabels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvLabels.npy')
    for lb in range(10):
        print('loading data',lb)
        Rec_clean[lb] = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet-Rec-{lb}.npy')
        Rec_adv[lb] = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet-Rec-{lb}.npy')
        for opt.interpret_method in interpret_methods:
            Z_clean[(lb,opt.interpret_method)] = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret-Rec-Z-{lb}.npy')
            Z_adv[(lb,opt.interpret_method)] = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret-Rec-Z-{lb}.npy')

    def test_inner(dataset,labels,phase):
        dataset = TensorDataset(dataset,labels)
        dataloader = DataLoader(dataset = dataset, batch_size=opt.val_batchsize,shuffle=False)
        correct = 0
        global_step = 0
        size = len(dataloader.dataset)
        # Loop over all examples in test set
        c = []
        for data, target in tqdm(dataloader, desc=phase):
            # Send the data and label to the device
            data, target = data.to(opt.device), target.to(opt.device)
            batch_size = data.shape[0]

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1)[1] # get the index of the max log-probability
            correct += torch.sum(init_pred == target.data)
            c.append(init_pred==target.data)
        print("{}_correct = {:.4f}".format(phase,correct.item()/size))
        c = torch.cat(c)
        return c
    
    def test_best(datasets,labels,phase):
        c = None
        for lb in range(10):
            ds = datasets[lb]
            tmp = test_inner(ds,labels,'test best {} {}'.format(phase,lb)).long()
            if c is None:
                c = tmp 
            else:
                c = c + tmp
        s = torch.sum(c)
        print(phase,'best correct = {:.4f}'.format(s.item()/c.shape[0]))
    # test_best(Rec_clean,CleanLabels,'clean')
    # test_best(Rec_adv,AdvLabels,'adv')
    # return 

    dataclean = []
    for i in range(Rec_clean[0].shape[0]):
        Zs = []
        for lb in range(10):
            Z = []
            for opt.interpret_method in interpret_methods:
                Z.append(Z_clean[(lb,opt.interpret_method)][i])
            # print(Z)
            Zs.append(min(Z))
        m = max(Zs)
        ind = Zs.index(m)
        dataclean.append(Rec_clean[ind][i:i+1])
    dataclean = torch.cat(dataclean)
    test_inner(dataclean,CleanLabels,'CleanRec')

    dataadv = []
    for i in range(Rec_adv[0].shape[0]):
        Zs = []
        for lb in range(10):
            Z = []
            for opt.interpret_method in interpret_methods:
                Z.append(Z_adv[(lb,opt.interpret_method)][i])
            Zs.append(min(Z))
        m = max(Zs)
        ind = Zs.index(m)
        dataadv.append(Rec_adv[ind][i:i+1])
    dataadv = torch.cat(dataadv)
    test_inner(dataadv,AdvLabels,'AdvRec')

def test_sum(interpret_methods = ['DL','LRP','VG','GBP','IG','org']):
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    Rec_clean={}
    Rec_adv = {}
    Z_adv = {}
    Z_clean = {}
    CleanLabels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanLabels.npy')
    AdvLabels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvLabels.npy')
    for lb in range(10):
        print('loading data',lb)
        Rec_clean[lb] = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet-Rec-{lb}.npy')
        Rec_adv[lb] = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet-Rec-{lb}.npy')
        for opt.interpret_method in interpret_methods:
            Z_clean[(lb,opt.interpret_method)] = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret-Rec-Z-{lb}.npy')
            Z_adv[(lb,opt.interpret_method)] = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret-Rec-Z-{lb}.npy')

    def test_inner(dataset,labels,phase):
        dataset = TensorDataset(dataset,labels)
        dataloader = DataLoader(dataset = dataset, batch_size=opt.val_batchsize,shuffle=False)
        correct = 0
        global_step = 0
        size = len(dataloader.dataset)
        # Loop over all examples in test set
        c = []
        for data, target in tqdm(dataloader, desc=phase):
            # Send the data and label to the device
            data, target = data.to(opt.device), target.to(opt.device)
            batch_size = data.shape[0]

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1)[1] # get the index of the max log-probability
            correct += torch.sum(init_pred == target.data)
            c.append(init_pred==target.data)
        print("{}_correct = {:.4f}".format(phase,correct.item()/size))
        c = torch.cat(c)
        return c

    dataclean = []
    for i in range(Rec_clean[0].shape[0]):
        Zs = []
        for lb in range(10):
            Z = []
            for opt.interpret_method in interpret_methods:
                Z.append(Z_clean[(lb,opt.interpret_method)][i])
            # print(Z)
            Zs.append(sum(Z))
        m = max(Zs)
        ind = Zs.index(m)
        dataclean.append(Rec_clean[ind][i:i+1])
    dataclean = torch.cat(dataclean)
    test_inner(dataclean,CleanLabels,'CleanRec')

    dataadv = []
    for i in range(Rec_adv[0].shape[0]):
        Zs = []
        for lb in range(10):
            Z = []
            for opt.interpret_method in interpret_methods:
                Z.append(Z_adv[(lb,opt.interpret_method)][i])
            Zs.append(sum(Z))
        m = max(Zs)
        ind = Zs.index(m)
        dataadv.append(Rec_adv[ind][i:i+1])
    dataadv = torch.cat(dataadv)
    test_inner(dataadv,AdvLabels,'AdvRec')

def test_th(ths,interpret_methods = ['DL','LRP','VG','GBP','IG','org']):
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    Rec_clean={}
    Rec_adv = {}
    Z_adv = {}
    Z_clean = {}
    CleanLabels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanLabels.npy')
    AdvLabels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvLabels.npy')
    for lb in range(10):
        print('loading data',lb)
        Rec_clean[lb] = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet-Rec-{lb}.npy')
        Rec_adv[lb] = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet-Rec-{lb}.npy')
        for opt.interpret_method in interpret_methods:
            Z_clean[(lb,opt.interpret_method)] = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret-Rec-Z-{lb}.npy')
            Z_adv[(lb,opt.interpret_method)] = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret-Rec-Z-{lb}.npy')

    def test_inner(dataset,labels,phase):
        dataset = TensorDataset(dataset,labels)
        dataloader = DataLoader(dataset = dataset, batch_size=opt.val_batchsize,shuffle=False)
        correct = 0
        global_step = 0
        size = len(dataloader.dataset)
        # Loop over all examples in test set
        c = []
        for data, target in tqdm(dataloader, desc=phase):
            # Send the data and label to the device
            data, target = data.to(opt.device), target.to(opt.device)
            batch_size = data.shape[0]

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1)[1] # get the index of the max log-probability
            correct += torch.sum(init_pred == target.data)
            c.append(init_pred==target.data)
        print("{}_correct = {:.4f}".format(phase,correct.item()/size))
        c = torch.cat(c)
        return c
    

    dataclean = []
    for i in range(Rec_clean[0].shape[0]):
        Zs = []
        for lb in range(10):
            Z = []
            for opt.interpret_method in interpret_methods:
                Z.append(Z_clean[(lb,opt.interpret_method)][i]>ths[opt.interpret_method])
            # print(Z)
            Zs.append(sum(Z))
        m = max(Zs)
        ind = Zs.index(m)
        dataclean.append(Rec_clean[ind][i:i+1])
    dataclean = torch.cat(dataclean)
    test_inner(dataclean,CleanLabels,'CleanRec')

    dataadv = []
    for i in range(Rec_adv[0].shape[0]):
        Zs = []
        for lb in range(10):
            Z = []
            for opt.interpret_method in interpret_methods:
                Z.append(Z_adv[(lb,opt.interpret_method)][i]>ths[opt.interpret_method])
            Zs.append(sum(Z))
        m = max(Zs)
        ind = Zs.index(m)
        dataadv.append(Rec_adv[ind][i:i+1])
    dataadv = torch.cat(dataadv)
    test_inner(dataadv,AdvLabels,'AdvRec')
            


def test_ref(interpret_methods = ['DL','LRP','VG','GBP','IG','org']):
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    Rec_clean={}
    Rec_adv = {}
    Z_adv = {}
    Z_clean = {}
    CleanLabels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanLabels.npy')
    AdvLabels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvLabels.npy')
    for lb in range(10):
        print('loading data',lb)
        Rec_clean[lb] = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet-Rec-{lb}.npy')
        Rec_adv[lb] = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet-Rec-{lb}.npy')
        for opt.interpret_method in interpret_methods:
            Z_clean[(lb,opt.interpret_method)] = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret-Rec-Z-{lb}.npy')
            Z_adv[(lb,opt.interpret_method)] = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret-Rec-Z-{lb}.npy')

    _clean,_adv = get_prev_z()

    def test_inner(dataset,labels,phase):
        dataset = TensorDataset(dataset,labels)
        dataloader = DataLoader(dataset = dataset, batch_size=opt.val_batchsize,shuffle=False)
        correct = 0
        global_step = 0
        size = len(dataloader.dataset)
        # Loop over all examples in test set
        c = []
        for data, target in tqdm(dataloader, desc=phase):
            # Send the data and label to the device
            data, target = data.to(opt.device), target.to(opt.device)
            batch_size = data.shape[0]

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1)[1] # get the index of the max log-probability
            correct += torch.sum(init_pred == target.data)
            c.append(init_pred==target.data)
        print("{}_correct = {:.4f}".format(phase,correct.item()/size))
        c = torch.cat(c)
        return c
    

    dataclean = []
    for i in range(Rec_clean[0].shape[0]):
        Zs = []
        for lb in range(10):
            prev_Z = []
            for opt.interpret_method in interpret_methods:
                prev_Z.append(_clean[opt.interpret_method][i])
            mm = min(prev_Z)
            _ind = prev_Z.index(mm)
            i_method = interpret_methods[_ind]
            Zs.append(Z_clean[(lb,i_method)][i])

        m = max(Zs)
        ind = Zs.index(m)
        dataclean.append(Rec_clean[ind][i:i+1])
    dataclean = torch.cat(dataclean)
    test_inner(dataclean,CleanLabels,'CleanRec')

    dataadv = []
    for i in range(Rec_adv[0].shape[0]):
        Zs = []
        for lb in range(10):
            prev_Z = []
            for opt.interpret_method in interpret_methods:
                prev_Z.append(_adv[opt.interpret_method][i])
            mm = min(prev_Z)
            _ind = prev_Z.index(mm)
            i_method = interpret_methods[_ind]
            Zs.append(Z_adv[(lb,i_method)][i])

        m = max(Zs)
        ind = Zs.index(m)
        dataadv.append(Rec_adv[ind][i:i+1])
    dataadv = torch.cat(dataadv)
    test_inner(dataadv,AdvLabels,'AdvRec')



def show_image():
    img = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/sprate_AdvDataSet-Rec-0/0.npy')
    # img = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/org/sprate_AdvInterpret/0.npy')
    img = torch.from_numpy(img).view(1,3,32,32)
    vutils.save_image(img,f'../{opt.tmp_dataset}/{opt.attack}_{opt.attack_param}_advimg.png')

    img = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/sprate_CleanDataSet-Rec-0/0.npy')
    # img = np.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/org/sprate_AdvInterpret/0.npy')
    img = torch.from_numpy(img).view(1,3,32,32)
    vutils.save_image(img,f'../{opt.tmp_dataset}/{opt.attack}_{opt.attack_param}_cleanimg.png')

if __name__ == '__main__':
    print("rectify.py")
    opt = Options().parse_arguments()
    t1 = datetime.now()
    opt.dataset = 'cifar10'
    # opt.dataset = 'fmnist'
    opt.gpu_id = 1
    
    opt.weight_decay=5e-4
    # opt.GAUS=0.05 # fmnist -> 0.1
    # opt.loss2_alp=0.5
    opt.num_epoches = 40
    opt.lr = 0.007
    opt.lr_step = 15
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
        opt.train_batchsize = opt.val_batchsize = 256
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


    # opt.attack = 'CW-U'
    # for param in [0.5,1,6,9,15]:
    #     opt.cw_confidence = opt.attack_param = param
    #     # generate()
    #     # generate_ensemble()
    #     train()
    #     test()

    opt.attack = 'PGD-U'
    first = True
    for param in [0.01,0.015,0.005,0.007,0.009,0.013]:
    # for param in [0.005]:
        opt.pgd_eps = opt.attack_param = param
        # generate()
        # generate_rectified()
        # show_image()
        # ths = test_ensemble_reducer()
        # get_Z()
        test()
        test_sum()
        # test_th(ths)
        test_ref()
