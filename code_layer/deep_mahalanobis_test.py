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
from functions import classify

warnings.filterwarnings("ignore")
from utils import *

NORMALIZE_IMAGES = {
    'mnist': ((0.1307,), (0.3081,)),
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'svhn': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))}

# NOISE_MAG_LIST = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
NOISE_MAG_LIST = [0.01,0.002,0.001]


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

def get_Mahalanobis_score(model, device, test_loader, num_classes, sample_mean, precision,
                          layer_index, magnitude,net_type='cifar10'):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    if model.training:
        model.eval()

    scale_images = NORMALIZE_IMAGES[net_type][1]
    n_channels = len(scale_images)
    Mahalanobis = []

    for data, target in tqdm(test_loader,desc = f'get mah at {layer_index}, mag={magnitude}'):
        data = data.to(device=device, dtype=torch.float)
        data.requires_grad = True
        # target = target.to(device=device)
        
        out_features = model.intermediate_forward(data, layer_index)
        sz = out_features.size()
        if len(sz) > 2:
            # Dimension reduction for the layer embedding
            # `N x C x H x W` tensor is converted to a `N x C` tensor by average pooling
            out_features = out_features.view(sz[0], sz[1], -1)
            out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features - batch_sample_mean
            #check if both parameters to multiplication are the same type
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad, 0)
        gradient = (gradient.float() - 0.5) * 2

        if n_channels == 1:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda(device=device)) / scale_images[0])
        else:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda(device=device)) / scale_images[0])
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda(device=device)) / scale_images[1])
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda(device=device)) / scale_images[2])

        with torch.no_grad():
            tempInputs = torch.add(data, -magnitude, gradient).to(device=device, dtype=torch.float)
            noise_out_features = model.intermediate_forward(tempInputs, layer_index)
            sz = noise_out_features.size()
            if len(sz) > 2:
                noise_out_features = noise_out_features.view(sz[0], sz[1], -1)
                noise_out_features = torch.mean(noise_out_features, 2)

        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.detach().cpu().numpy())

    return Mahalanobis

def sample_estimator(model, device, num_classes, layer_dimension_reduced, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    if model.training:
        model.eval()

    correct, total = 0, 0
    num_output = len(layer_dimension_reduced)   # number of layers
    num_sample_per_class = np.zeros(num_classes, dtype=np.int)
    # `list_features` is a list of length equal to the number of layers; each item is a list of 0s of length
    # number of classes
    list_features = []
    for _ in range(num_output):
        list_features.append([0] * num_classes)

    with torch.no_grad():
        for data, target in tqdm(train_loader,desc='estimator'):
            data = data.to(device)
            target = target.to(device)
            n_batch = data.size(0)
            total += n_batch

            # Get the intermediate layer embeddings and the DNN output
            output, out_features = model.layer_wise_deep_mahalanobis(data)
            # Dimension reduction for the layer embeddings.
            # Each `N x C x H x W` tensor is converted to a `N x C` tensor by average pooling
            for i in range(num_output):
                sz = out_features[i].size()
                if len(sz) > 2:
                    out_features[i] = out_features[i].view(sz[0], sz[1], -1)
                    out_features[i] = torch.mean(out_features[i], 2)
                    # print("la:", out_features[i].shape)

            # compute the accuracy
            pred = output.max(1)[1]
            correct += pred.eq(target).sum().item()

            # construct the sample matrix for each layer and each class
            for i in range(n_batch):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = \
                            torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1

                num_sample_per_class[label] += 1

    # `list_features` will be a list of list of torch tensors. The first list indexes the layers and the second list
    # indexes the classes. Each tensor has samples from a particular layer and a particular class
    # print(num_sample_per_class)
    # for i in range(num_output):
    #     for j in range(num_classes):
    #         print(i, j, list_features[i][j].shape)

    # Compute the sample mean for each layer and each class
    sample_class_mean = []
    out_count = 0
    for num_feature in layer_dimension_reduced:
        temp_list = torch.zeros(num_classes, num_feature).to(device)
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)

        sample_class_mean.append(temp_list)
        out_count += 1

    '''
    print("sample_class_mean")
    for i in range(num_output):
        for j in range(num_classes):
            print(i, j, sample_class_mean[i][j].shape)
    '''

    # Sample inverse covariance matrix estimation for each layer with data from all the classes combined
    # (i.e. a shared inverse covariance matrix)
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    precision = []
    for k in range(num_output):
        X = list_features[k][0] - sample_class_mean[k][0]
        for i in range(1, num_classes):
            X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        group_lasso.fit(X.detach().cpu().numpy())
        temp_precision = torch.from_numpy(group_lasso.precision_).to(dtype=torch.float, device=device)
        precision.append(temp_precision)

    # `precision` will be a list of torch tensors with the precision matrix per layer
    '''
    print("precision")
    for i in range(num_output):
        print(i, precision[i].shape)
    '''
    print('\n Accuracy:({:.4f}%)\n'.format(100. * correct / total))
    return sample_class_mean, precision


def generate_mahalanobis(CleanDataSet,AdvDataSet,model,num_layers,sample_mean,precision,noise_mag):
    num_labels=opt.classifier_classes
    def generate_once(dataset):
        mahalanobis_feat = None
        loader = TensorDataset(dataset,torch.ones(dataset.shape[0]).long())
        loader = DataLoader(dataset = loader, batch_size=opt.val_batchsize,shuffle=True)
        for i in range(num_layers):
            M_in = get_Mahalanobis_score(
                model, opt.device, loader, num_labels, sample_mean, precision, i, noise_mag
            )
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                mahalanobis_feat = M_in.reshape((M_in.shape[0], -1))
            else:
                mahalanobis_feat = np.concatenate((mahalanobis_feat, M_in.reshape((M_in.shape[0], -1))), axis=1)

        # output array has shape `(n_samples, n_layers)`
        return np.asarray(mahalanobis_feat, dtype=np.float32)
    
    CleanMah = generate_once(CleanDataSet)
    AdvMah = generate_once(AdvDataSet)

    np.save(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/CleanMah-{noise_mag}.npy',CleanMah)
    np.save(f'../{opt.tmp_dataset}/{opt.data_phase}/{opt.attack}_{opt.attack_param}/AdvMah-{noise_mag}.npy',AdvMah)

    return CleanMah,AdvMah


def train_logistic_classifier(features,labels, n_cv_folds=5,
                              scale_features=False, balance_classes=False,
                              n_jobs=-1, max_iter=200, seed_rng=123):

    lab_uniq, count_classes = np.unique(labels, return_counts=True)
    if len(lab_uniq) == 2 and lab_uniq[0] == 0 and lab_uniq[1] == 1:
        pass
    else:
        raise ValueError("Did not receive expected binary class labels 0 and 1.")

    pos_prop = float(count_classes[1]) / (count_classes[0] + count_classes[1])
    if scale_features:
        # Min-max scaling to preprocess all features to the same range [0, 1]
        scaler = MinMaxScaler().fit(features)
        features = scaler.transform(features)
    else:
        scaler = None

    print("\nTraining a binary logistic classifier with {:d} samples and {:d} Mahalanobis features.".
          format(*features.shape))
    print("Using {:d}-fold cross-validation with area under ROC curve as the metric to select the best "
          "regularization hyperparameter.".format(n_cv_folds))
    print("Proportion of positive (adversarial or OOD) samples in the training data: {:.4f}".format(pos_prop))
    if pos_prop <= 0.1:
        # high imbalance in the classes
        balance_classes = True

    class_weight = None
    if balance_classes:
        if (pos_prop < 0.45) or (pos_prop > 0.55):
            class_weight = {0: 1.0 / (1 - pos_prop),
                            1: 1.0 / pos_prop}
            print("Balancing the classes by assigning sample weight {:.4f} to class 0 and sample weight {:.4f} "
                  "to class 1".format(class_weight[0], class_weight[1]))

    model_logistic = LogisticRegressionCV(
        cv=n_cv_folds,
        penalty='l2',
        scoring='roc_auc',
        multi_class='auto',
        class_weight=class_weight,
        max_iter=max_iter,
        refit=True,
        n_jobs=n_jobs,
        random_state=seed_rng
    ).fit(features, labels)

    # regularization coefficient values
    coeffs = model_logistic.Cs_
    # regularization coefficient corresponding to the maximum cross-validated AUC
    coeff_best = model_logistic.C_[0]
    mask = np.abs(coeffs - coeff_best) < 1e-16
    ind = np.where(mask)[0][0]
    # average AUC across the test folds for the best regularization coefficient
    auc_scores = model_logistic.scores_[1]      # has shape `(n_cv_folds, coeffs.shape[0])`
    auc_avg_best = np.mean(auc_scores[:, ind])
    print("Average AUC from the test folds: {:.6f}".format(auc_avg_best))

    # proba = model_logistic.predict_proba(features)[:, -1]
    model_dict = {'logistic': model_logistic,
                  'scaler': scaler,
                  'auc_avg': auc_avg_best}
    return model_dict


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


def fit_detector(CleanDataSet,AdvDataSet,model,num_layers,sample_mean,precision):
    num_labels=opt.classifier_classes
    # Noise magnitude values
    noise_mag_list = NOISE_MAG_LIST
    model_dict_best = {}
    noise_mag_best = NOISE_MAG_LIST[0]
    auc_max = -1.
    for magnitude in noise_mag_list:
        print('\nNoise: ' + str(magnitude))
        CleanMah,AdvMah = generate_mahalanobis(CleanDataSet,AdvDataSet,model,num_layers,sample_mean,precision,magnitude)
        features,labels = merge_and_generate_labels(CleanMah,AdvMah)

        print("Training a logistic classifier to discriminate in-distribution from OOD/adversarial samples")
        model_dict_curr = train_logistic_classifier(features,labels, scale_features=False, balance_classes=False,
                                                    n_jobs=opt.workers)
        auc_curr = model_dict_curr['auc_avg']
        if auc_curr > auc_max:
            auc_max = auc_curr
            model_dict_best = model_dict_curr
            noise_mag_best = magnitude

    print("\nNoise magnitude {:.6f} resulted in the logistic classifier with maximum average AUC = {:.6f}".
          format(noise_mag_best, auc_max))
    model_detector = model_dict_best
    model_detector['sample_mean'] = sample_mean
    model_detector['precision'] = precision
    model_detector['noise_magnitude'] = noise_mag_best
    model_detector['n_classes'] = num_labels
    model_detector['n_layers'] = num_layers

    return model_detector

def test_detector(CleanDataSet,AdvDataSet,model,num_layers,sample_mean,precision,model_detector):
    num_labels=opt.classifier_classes
    # Noise magnitude values
    noise_mag_list = NOISE_MAG_LIST
    model_dict_best = {}
    noise_mag_best = NOISE_MAG_LIST[0]
    auc_max = -1.
    magnitude = model_detector['noise_magnitude']
    CleanMah,AdvMah = generate_mahalanobis(CleanDataSet,AdvDataSet,model,num_layers,sample_mean,precision,magnitude)
    features,labels = merge_and_generate_labels(CleanMah,AdvMah)

    logistic = model_detector['logistic']
    y_pred = logistic.predict_proba(features)
    y_pred = y_pred[:,1]
    roc_auc,threshold,thresholds = auc_curve(labels,y_pred,plt=False)

    return roc_auc


def main():
    num_labels=opt.classifier_classes
    opt.data_phase='train'
    CleanDataSet,AdvDataSet,CleanLabels,AdvLabels,Labels = get_sift_data()

    # merge loader
    ds = torch.cat([CleanDataSet,AdvDataSet])
    lb = torch.cat([CleanLabels,AdvLabels]).long()
    size = lb.shape[0]
    loader = TensorDataset(ds,lb)
    loader = DataLoader(dataset = loader, batch_size=opt.val_batchsize,shuffle=True)

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

    # shape
    size_data = CleanDataSet.size()
    temp_x = torch.rand(2, size_data[1], size_data[2], size_data[3]).to(opt.device)
    _, temp_list = model.layer_wise_deep_mahalanobis(temp_x)
    num_layers = len(temp_list)

    feature_list = np.zeros(num_layers, dtype=np.int)
    for i, out in enumerate(temp_list):
        feature_list[i] = out.size(1)   # num. channels for conv. layers; num. dimensions for FC layers
    
    sample_mean, precision = sample_estimator(model, opt.device, num_labels, feature_list, loader_clean)

    # fit
    model_detector = fit_detector(CleanDataSet,AdvDataSet,model,num_layers,sample_mean,precision)

    # test
    opt.data_phase='test'
    CleanDataSet,AdvDataSet,CleanLabels,AdvLabels,Labels = get_sift_data()
    auc = test_detector(CleanDataSet,AdvDataSet,model,num_layers,sample_mean,precision,model_detector)
    print(f"deep mahalanobis test for attack = {opt.attack} param = {opt.attack_param}, auc score = {auc}")
    return auc



if __name__ == '__main__':
    print("deep_mahalanobis_test.py")
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

    opt.classifier_net = 'cwnet'
    # opt.device = torch.device("cuda:{}".format(opt.gpu_id))
    opt.interpret_method = 'VG'
    
    opt.detect_method = 'iforest' # ['iforest','SVDD','LOF','Envelope'] # Envelope 太难跑
    opt.minus = False
    opt.scale = 1
    report_args(opt)


    opt.data_phase='train'
    # opt.attack = 'PGD-T'
    opt.attack_param=0.007
    opt.max_num=10000
    classify(opt)
    generate()
    generate_result_only()
    raise NotImplemented('yahahah')

    auc_dict = {}
    # for opt.classifier_net,opt.dataset,opt.image_channels in [('vgg11bn','cifar10',3)]:
    # for opt.classifier_net,opt.dataset,opt.image_channels in [('wide_resnet_small','imagenet',3)]:
    for opt.classifier_net,opt.dataset,opt.image_channels in [('cwnet','cifar10',3)]:
        if "imagenet" in opt.dataset:
            opt.classifier_classes=30
        opt.tmp_dataset = f'tmp_dataset_{opt.classifier_net}_{opt.dataset}'
        auc_values = {}
        auc_values_train = {}
        # for opt.attack,opt.attack_param in [("PGD-T",0.015),("PGD-U",0.015),("DDN-U",1000),("DDN-T",1000),("CW-T",9),("CW-U",9),("NES-U",0),("Optim-U",0),("Boundary-U",0)]:
        # for opt.attack,opt.attack_param in [('CW-T',9),('CW-U',9),('PGD-T',0.007),('PGD-U',0.007)]:
        for opt.attack,opt.attack_param in [("CW-T",9),("CW-U",9)]:
            auc = main()
            auc_dict[opt.attack] = {'deep_mah':auc}
        print(auc_dict)
        timeStamp = datetime.now()
        formatTime = timeStamp.strftime("%m-%d %H-%M-%S")
        auc_dict = pd.DataFrame.from_dict(auc_dict)
        auc_dict.to_csv(f'../{opt.tmp_dataset}/attack_eps/deep_mah-{formatTime}.csv',float_format = '%.3f')

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