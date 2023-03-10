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

    make_dir(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}')

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

    for v in ['CleanResult','AdvResult']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')
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

    make_dir(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}')

        
    for v in ['CleanDataSet','AdvDataSet','Labels']:
        n = eval(v)
        n = torch.cat(n,dim=0)
        torch.save(n,f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{v}.npy')

    return

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
                break
    
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

    def source(self,ensemble_method):
        if(ensemble_method == 'Min'):
            Z = []
            X = []
            S = []
            for ind in range(length):
                min = 1e10
                for k,r in self.result.items():
                    if r[ind]<min:
                        min = r[ind]
                        s = k
                Z.append(min)
                S.append(s)
            Z = np.array(Z)
        else:
            raise Exception(f"{self.ensemble_method} not implemented")

        return X


def generate_ensemble(interpret_methods=['DL','LRP','VG','GBP','IG']):
    generate_result_only()
    CleanDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    CleanResult = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanResult.npy')
    AdvResult = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvResult.npy')

    CleanDataSet = CleanDataSet[CleanResult==Labels]
    CleanLabels = Labels[CleanResult==Labels]
    AdvDataSet = AdvDataSet[(CleanResult==Labels)&(AdvResult!=Labels)]
    AdvLabels = Labels[(CleanResult==Labels)&(AdvResult!=Labels)]

    torch.save(CleanLabels,f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanLabels.npy')
    torch.save(AdvLabels,f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvLabels.npy')

    reducers = {}
    ensembled_clf = Ensembler(interpret_methods=interpret_methods,detect_method='iforest')

    for opt.interpret_method in interpret_methods:
        print('train',opt.interpret_method)
        if opt.interpret_method=='org':
            CleanInterpret = CleanDataSet
            AdvInterpret = AdvDataSet
        else:
            # generate_interpret_only()
            try:
                CleanInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
                AdvInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')
            except:
                generate_interpret_only()
                CleanInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/CleanInterpret.npy')
                AdvInterpret = torch.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/AdvInterpret.npy')
            CleanInterpret = CleanInterpret[CleanResult==Labels]
            AdvInterpret = AdvInterpret[(CleanResult==Labels)&(AdvResult!=Labels)]

        X_train = CleanInterpret.clone().cpu().view(CleanInterpret.shape[0],-1).numpy()
        reducer = PCA_Reducer()
        reducer.train(X_train)
        reducers[opt.interpret_method] = reducer
        ensembled_clf.fit(opt.interpret_method,X_train)


        # print(CleanInterpret.shape[0],AdvInterpret.shape[0])
        # make_dir(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/sprate_CleanInterpret')
        # for ind in tqdm(range(CleanInterpret.shape[0])):
        #     item = CleanInterpret[ind].numpy()
        #     np.save(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/sprate_CleanInterpret/{ind}.npy',item)
        # make_dir(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/sprate_AdvInterpret')
        # for ind in tqdm(range(AdvInterpret.shape[0])):
        #     item = AdvInterpret[ind].numpy()
        #     np.save(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/sprate_AdvInterpret/{ind}.npy',item)

    make_dir(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/rec_CleanInterpret')
    for ind in tqdm(range(CleanDataSet.shape[0])):
        Xmap = {}
        Omap = {}
        img = CleanDataSet[ind].numpy()
        for opt.interpret_method in interpret_methods:
            item = np.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/sprate_CleanInterpret/{ind}.npy')       
            Omap[opt.interpret_method] = item
            item = torch.from_numpy(item).view(1,-1)
            item = item.numpy()
            item = reducers[opt.interpret_method].reduce(item)
            Xmap[opt.interpret_method] = item
        result,result_method = ensembled_clf.calculate_once_min(Xmap)
        if(result_method == 'org'):
            raise Exception
        else:
            # item = Omap[result_method]
            # th = np.percentile(item,opt.rec_percent)
            # img = img*(item<th)
            if opt.mask_method == 'random':
                mean,std = np.mean(img),np.std(img)
                value = np.random.normal(mean,std,img.shape)
            else:
                value = 0
            item = Omap[result_method]
            mask = np.max(item,axis=0)
            # mask = item
            th = np.percentile(mask,opt.rec_percent)
            mask = (mask<th)
            mask = np.repeat(np.expand_dims(mask,0),img.shape[0],0)
            img = img*mask + value*(1-mask)
            img = np_filter(img)
            np.save(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/rec_CleanInterpret/{ind}.npy',img)

    make_dir(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/rec_AdvInterpret')
    for ind in tqdm(range(AdvDataSet.shape[0])):
        Xmap = {}
        Omap = {}
        img = AdvDataSet[ind].numpy()
        for opt.interpret_method in interpret_methods:
            item = np.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/{opt.interpret_method}/sprate_AdvInterpret/{ind}.npy')       
            Omap[opt.interpret_method] = item
            item = torch.from_numpy(item).view(1,-1)
            item = item.numpy()
            item = reducers[opt.interpret_method].reduce(item)
            Xmap[opt.interpret_method] = item
        result,result_method = ensembled_clf.calculate_once_min(Xmap)
        if(result_method == 'org'):
            raise Exception
        else:
            if opt.mask_method == 'random':
                mean,std = np.mean(img),np.std(img)
                value = np.random.normal(mean,std,img.shape)
            else:
                value = 0
            item = Omap[result_method]
            mask = np.max(item,axis=0)
            # mask = item
            th = np.percentile(mask,opt.rec_percent)
            mask = (mask<th)
            mask = np.repeat(np.expand_dims(mask,0),img.shape[0],0)
            img = img*mask + value*(1-mask)
            img = np_filter(img)
            np.save(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/rec_AdvInterpret/{ind}.npy',img)
            # im = Image.fromarray(img)
            # im.save(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/rec_AdvInterpret/{ind}.jpg')

                



# def get_ensembler(interpret_methods=['DL','LRP','org','VG','GBP','IG'],train=True):
#     reducer = PCA_Reducer()
#     ensembled_clf = Ensembler(interpret_methods=interpret_methods,detect_method='iforest')
#     for opt.interpret_method in interpret_methods:
#         generate_interpret_only()
#         CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
#         cleansize = CleanInterpret.shape[0]
#         advsize = AdvInterpret.shape[0]
#         total = Labels.shape[0]
#         print(f"i_method is {opt.interpret_method}, total is {total}, clean size is {cleansize} (clean/total={cleansize/total}), adv size is {advsize} (adv/clean={advsize/cleansize})")

#         clf_path_name = f'../classifier_pth/{opt.detect_method}_clf-ensemble'
#         if train:
#             print('training begin')
#             t1 = datetime.now()
#             X_train = CleanInterpret.view(cleansize,-1).numpy()
#             X_train = reducer.train(X_train)
#             ensembled_clf.fit(opt.interpret_method,X_train)
#             t2 = datetime.now()
#             totalseconds = (t2-t1).total_seconds()
#             h = totalseconds // 3600
#             m = (totalseconds - h*3600) // 60
#             s = totalseconds - h*3600 - m*60
#             print('training end in time {}h {}m {:.4f}s'.format(h,m,s))
#             ensembled_clf.save(clf_path_name)
#         else:
#             X_train = CleanInterpret.view(cleansize,-1).numpy()
#             X_train = reducer.train(X_train)
#             ensembled_clf.load(clf_path_name)
#     return ensembled_clf,reducer


def train():
    datapath = f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/rec_CleanInterpret'
    labelpath = f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanLabels.npy'
    datapath2 = f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/rec_AdvInterpret'
    labelpath2 = f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvLabels.npy'
    dataloader = seprate_dataset([datapath,datapath2],[labelpath,labelpath2],opt.train_batchsize,shuffle=True,num_workers=opt.workers)

    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('using network {}'.format(opt.classifier_net))
    saved_name = f'../classifier_pth/rectifynet_{opt.classifier_net}_{opt.dataset}_{opt.mask_method}_{opt.rec_percent}.pth'
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.lr_mom, weight_decay=5e-4)    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
    criterion = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    model.to(opt.device)
    model.train()

    for epoch in range(opt.num_epoches):
        print('Epoch {}/{}'.format(epoch+1, opt.num_epoches))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data
        
        for inputs, labels in tqdm(dataloader, desc='train'):
            inputs = inputs.float().to(opt.device)
            labels = labels.to(opt.device)
            optimizer.zero_grad()

            outputs = model(inputs)
            preds = outputs.max(1)[1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / size
        epoch_acc = running_corrects.double() / size
        scheduler.step()
        
        # if epoch % 10 == 0:
        #     model.step_k()

        print('train Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        torch.save(model.state_dict(), saved_name)

def test():
    datapath = f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/rec_AdvInterpret'
    labelpath = f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvLabels.npy'
    dataloader = seprate_dataset([datapath],[labelpath],opt.train_batchsize,shuffle=True,num_workers=opt.workers)

    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('using network {}'.format(opt.classifier_net))
    saved_name = f'../classifier_pth/rectifynet_{opt.classifier_net}_{opt.dataset}_{opt.mask_method}_{opt.rec_percent}.pth'
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()

    acc = 0
    size = len(dataloader.dataset)
    for inputs, labels in tqdm(dataloader, desc='train'):
        inputs = inputs.float().to(opt.device)
        labels = labels.to(opt.device)

        outputs = model(inputs)
        preds = outputs.max(1)[1]
        acc += torch.sum(preds == labels.data)

    acc = acc.double() / size
    
    print(f'test {opt.attack}_{opt.attack_param} acc = {acc}')
        
def np_filter(img):
    def filter_once(posi,posj):
        r = np.array([0,0,0])
        c = 0
        for i in range(posi-1,posi+2):
            for j in range(posj-1,posj+2):
                if i<0 or i>=img.shape[1]:
                    continue
                if j<0 or j >= img.shape[2]:
                    continue
                r = r + img[:,i,j]
                c = c + 1
        r = r/c 
        return r
    for ii in range(img.shape[1]):
        for jj in range(img.shape[2]):
            img[:,ii,jj] = filter_once(ii,jj)
    return img

def show_image():
    img = np.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/rec_AdvInterpret/0.npy')
    # img = np.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/org/sprate_AdvInterpret/0.npy')
    img = torch.from_numpy(img).view(1,3,32,32)
    vutils.save_image(img,f'../tmp_dataset/{opt.attack}_{opt.attack_param}_advimg.png')

    img = np.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/rec_CleanInterpret/0.npy')
    # img = np.load(f'../tmp_dataset/{opt.data_phase}/{opt.attack}_{opt.attack_param}/org/sprate_AdvInterpret/0.npy')
    img = torch.from_numpy(img).view(1,3,32,32)
    vutils.save_image(img,f'../tmp_dataset/{opt.attack}_{opt.attack_param}_cleanimg.png')

if __name__ == '__main__':
    print("rectify.py")
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

    opt.mask_method = 'null'
    opt.rec_percent = 80

    # opt.attack = 'CW-U'
    # for param in [0.5,1,6,9,15]:
    #     opt.cw_confidence = opt.attack_param = param
    #     # generate()
    #     # generate_ensemble()
    #     train()
    #     test()

    opt.attack = 'PGD-U'
    first = True
    for param in [0.015,0.005,0.007,0.009,0.01,0.013]:
        opt.pgd_eps = opt.attack_param = param
        # generate()
        generate_ensemble()
        show_image()
        if first:
            train()
            test()
            first=False
        else:
            # generate()
            generate_ensemble()
            show_image()
            test()