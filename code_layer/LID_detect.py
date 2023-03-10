
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor,KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from detectors.detector_deep_knn import DeepKNN
from detectors.detector_lid_paper import DetectorLID,DetectorLIDClassCond,DetectorLIDBatch
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

def get_aug(data,eps):
    return data + torch.randn(data.size()).cuda(device=opt.device) * eps

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

def get_difference(p):
    CleanDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanDataSet.npy')
    AdvDataSet = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvDataSet.npy')
    Labels = torch.load(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/Labels.npy')
    x = CleanDataSet-AdvDataSet
    x = x.view(x.shape[0],-1)
    l = torch.norm(x,dim=1,p=p).mean().item()
    return l


def get_layers_embedding(dataloader,model,num_layers):
    layers_embedding = []
    for layer_index in range(num_layers):
        tmp = []
        for data, in tqdm(dataloader,desc = f'{opt.attack},get embeddings at layer-{layer_index}'):
            data = data.to(device=opt.device, dtype=torch.float)
            out_features = model.intermediate_forward(data, layer_index)
            shape = out_features.shape
            assert shape[0] == data.shape[0]
            tmp.append(out_features.detach().clone().cpu().view(shape[0],-1))
        tmp = torch.cat(tmp)
        layers_embedding.append(tmp)
    return layers_embedding


def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

class LID_helper:
    def __init__(self,model):
        batch_lid = False
        kwargs = {
            'n_neighbors': opt.knn_k,
            'skip_dim_reduction': True,
            'model_dim_reduction': None,
            'max_iter': 200,
            'balanced_classification': True,
            'n_jobs': opt.workers,
            'save_knn_indices_to_file': False,
            # 'seed_rng': 400
        }
        if batch_lid:
            det_model = DetectorLIDBatch(n_batches=10, **kwargs)
        else:
            det_model = DetectorLID(**kwargs)

        self.detector = det_model
        self.model = model
        self.model.eval()

    # must be clean examples
    def pack_train(self,CleanDataSet,AdvDataSet,num_layers):
        loader = TensorDataset(CleanDataSet)
        loader = DataLoader(dataset = loader, batch_size=opt.val_batchsize,shuffle=True)
        layers_embedding = get_layers_embedding(loader,self.model,num_layers)
        loader = TensorDataset(AdvDataSet)
        loader = DataLoader(dataset = loader, batch_size=opt.val_batchsize,shuffle=True)
        layers_embedding_adv = get_layers_embedding(loader,self.model,num_layers)
        self.detector.fit(layers_embedding,layers_embedding_adv)
    
    def pack_test(self,DataMap,num_layers,cleanup=False):
        loader = TensorDataset(DataMap)
        loader = DataLoader(dataset = loader, batch_size=opt.val_batchsize,shuffle=True)
        layers_embedding = get_layers_embedding(loader,self.model,num_layers)

        # z = self.detector.score(layers_embedding, cleanup=cleanup)
        z = self.detector.score(layers_embedding,cleanup=False)
        
        return z

    def dump(self,pathname):
        try:
            os.makedirs(pathname)
        except:
            print(f"{pathname} already exist")
        pk_dump(self.detector,f'{pathname}/dknn.npy')

    def load(self,pathname):
        self.detector = pk_load(f'{pathname}/dknn.npy')
        return self


def main():
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

    opt.data_phase='train'
    CleanDataSet,AdvDataSet,CleanLabels,AdvLabels,Labels = get_sift_data()

    # shape
    size_data = CleanDataSet.size()
    temp_x = torch.rand(2, size_data[1], size_data[2], size_data[3]).to(opt.device)
    _, temp_list = model.layer_wise_deep_mahalanobis(temp_x)
    num_layers = len(temp_list)

    feature_list = np.zeros(num_layers, dtype=np.int)
    for i, out in enumerate(temp_list):
        feature_list[i] = out.size(1)   # num. channels for conv. layers; num. dimensions for FC layers

    clf = LID_helper(model)
    clf.pack_train(CleanDataSet,AdvDataSet,num_layers)

    # eval
    opt.data_phase = 'train'
    CleanDataSet,AdvDataSet,CleanLabels,AdvLabels,Labels = get_sift_data()
    Z1 = clf.pack_test(CleanDataSet,num_layers,cleanup=False)
    Z2 = clf.pack_test(AdvDataSet,num_layers,cleanup=True)
    cleansize = CleanDataSet.shape[0]
    advsize = AdvDataSet.shape[0]
    # Z = np.concatenate([Z1,Z2],axis=0)
    # Y = torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy()
    Z,Y = merge_and_generate_labels(Z1,Z2)
    roc_auc,threshold,thresholds = auc_curve(Y,Z)
    acc = (Z[:cleansize]>threshold).sum() + (Z[cleansize:]>=threshold).sum()
    acc = acc/(cleansize+advsize)

    print(f"lid eval for attack = {opt.attack} param = {opt.attack_param}, auc score = {roc_auc}, acc = {acc}")

    # test
    opt.data_phase='test'
    CleanDataSet,AdvDataSet,CleanLabels,AdvLabels,Labels = get_sift_data()
    Z1 = clf.pack_test(CleanDataSet,num_layers)
    Z2 = clf.pack_test(AdvDataSet,num_layers)
    cleansize = CleanDataSet.shape[0]
    advsize = AdvDataSet.shape[0]
    # Z = np.concatenate([Z1,Z2],axis=0)
    # Y = torch.cat([torch.ones(cleansize),torch.zeros(advsize)],dim=0).numpy()
    Z,Y = merge_and_generate_labels(Z1,Z2)
    roc_auc,threshold,thresholds = auc_curve(Y,Z)
    acc = (Z[:cleansize]>threshold).sum() + (Z[cleansize:]>=threshold).sum()
    acc = acc/(cleansize+advsize)

    print(f"lid test for attack = {opt.attack} param = {opt.attack_param}, auc score = {roc_auc}, acc = {acc}")
    return roc_auc
    

if __name__ == '__main__':
    print("LID_test.py")
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
    opt.interpret_method = 'VG'
    
    opt.detect_method = 'iforest' # ['iforest','SVDD','LOF','Envelope'] # Envelope 太难跑
    opt.minus = False
    opt.scale = 1
    report_args(opt)



    auc_dict = {}
    opt.knn_k = None
    # for opt.classifier_net,opt.dataset,opt.image_channels in [('vgg11bn','cifar10',3)]:
    for opt.classifier_net,opt.dataset,opt.image_channels in [('wide_resnet_small','imagenet',3)]:
    # for opt.classifier_net,opt.dataset,opt.image_channels in [('cwnet','cifar10',3)]:
        if "imagenet" in opt.dataset:
            opt.classifier_classes=30
        opt.tmp_dataset = f'tmp_dataset_{opt.classifier_net}_{opt.dataset}'
        auc_values = {}
        auc_values_train = {}
        # for opt.attack,opt.attack_param in [("PGD-T",0.015),("PGD-U",0.015),("DDN-U",1000),("DDN-T",1000),("CW-T",9),("CW-U",9),("NES-U",0),("Optim-U",0),("Boundary-U",0)]:
        for opt.attack,opt.attack_param in [('CW-T',9),('CW-U',9),('PGD-T',0.007),('PGD-U',0.007)]:
        # for opt.attack,opt.attack_param in [("CW-T",9),("CW-U",9)]:
            auc = main()
            auc_dict[opt.attack] = {'lid':auc}
        print(auc_dict)
        timeStamp = datetime.now()
        formatTime = timeStamp.strftime("%m-%d %H-%M-%S")
        auc_dict = pd.DataFrame.from_dict(auc_dict)
        auc_dict.to_csv(f'../{opt.tmp_dataset}/attack_eps/lid-{formatTime}.csv',float_format = '%.3f')

    # auc_values = {}
    # opt.data_phase = 'train'
    # opt.max_num = 10000
    # opt.detector_train_epoch = 20
    # opt.detector_train_lr = 0.0001
    # for opt.attack in [ 'CW-T']:
    #     auc_values = {}
    #     for opt.cw_confidence in [9]:
    #         opt.attack_param = opt.cw_confidence
    #         generate()
    #         CleanDataSet,AdvDataSet,CleanInterpret,AdvInterpret,Labels = get_sift_data()
    #         attack_rate = AdvInterpret.shape[0]/CleanInterpret.shape[0]
    #         item = {'attack_rate':attack_rate}
    #         l2 = get_difference(2)
    #         item['l2'] = l2
    #         for opt.interpret_method in ['DL','LRP','org','VG','GBP','IG']:
    #             print("+"*10)
    #             generate_interpret_only()
    #             dnn_acc = train_detector_only()
    #             auc,acc = main_once(train=True)
    #             item[opt.interpret_method] = auc
    #             item[f"dnn_acc_{opt.interpret_method}"] = dnn_acc
    #             item[f"acc_{opt.interpret_method}"] = acc
    #             print('attack is {}, param is {:.4f}, attack rate is {:.4f}, l2 is {:.4f}, interpret is {}, auc is {:.4f}'.format(
    #                 opt.attack,opt.cw_confidence,attack_rate,l2,opt.interpret_method,auc
    #             ))
    #         auc_values[opt.cw_confidence] = item
    #         auc_sum, auc_breadth,auc_min,auc_max = main(train=False,interpret_methods = ['DL','LRP','org','VG','GBP','IG'])
    #         item['ensemble_sum_up'] = auc_sum
    #         item['ensemble_breadth_first'] = auc_breadth
    #         item['ensemble_min'] = auc_min
    #         item['ensemble_max'] = auc_max
    #     print(auc_values)
    #     pk_dump(auc_values,f'../{opt.tmp_dataset}/attack_eps_train/{opt.attack}.pth')
    #     auc_values = pk_load(f'../{opt.tmp_dataset}/attack_eps_train/{opt.attack}.pth')
    #     # plot_auc_eps(auc_values,f'../{opt.tmp_dataset}/attack_eps_train/fig-{opt.attack}.jpg',methods=['DL','LRP','org','VG','GBP','IG', 'ensemble_sum_up','ensemble_breadth_first','ensemble_min','ensemble_max','attack_rate'])
    #     auc_values = pd.DataFrame.from_dict(auc_values)
    #     auc_values.to_csv(f'../{opt.tmp_dataset}/attack_eps_train/{opt.attack}.csv',float_format = '%.3f')