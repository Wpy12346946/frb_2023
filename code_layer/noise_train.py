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
from attack_methods import adversarialattack
import re

os.environ['CUDA_VISIBLE_DEVICES']='0,1'

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

def train(opt):
    writer = SummaryWriter(opt.summary_name)
    train_loader = build_loader(opt.data_root, opt.dataset,\
     opt.train_batchsize, train=True, workers=opt.workers)

    # Initialize the network
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    # model = clf11bn(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)

    print('using network {}'.format(opt.classifier_net))
    # print('loading from {}'.format(saved_name))
    # model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.lr_mom, weight_decay=5e-4)    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
    criterion = nn.CrossEntropyLoss()
    size = len(train_loader.dataset)
    model.train()
    i=0
    since = time.time()
    for epoch in range(opt.num_epoches):
        print('Epoch {}/{}'.format(epoch+1, opt.num_epoches))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data
        
        for inputs, labels in tqdm(train_loader, desc='train'):
            inputs = inputs.to(opt.device)
            perturbed_inputs = inputs
            if opt.noise:
                perturbed_inputs = get_aug(inputs,eps=0.05)
            labels = labels.to(opt.device)
            optimizer.zero_grad()

            # outputs = model(perturbed_inputs)
            # _,org_list,_ = model.forward_layer(inputs)
            outputs,layer_list,layer_names = model.forward_layer(perturbed_inputs)
            preds = outputs.max(1)[1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss',loss.item(),global_step=i)
            writer.add_scalar('acc',torch.sum(preds==labels.data).item(),global_step=i)
            i=i+1
            
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
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    opt = Options().parse_arguments()
    t1 = datetime.now()
    opt.dataset = 'cifar10'
    # opt.dataset = 'fmnist'
    opt.gpu_id = 1
    
    opt.weight_decay=5e-4
    # opt.GAUS=0.05 # fmnist -> 0.1
    # opt.loss2_alp=0.5
    opt.num_epoches = 70
    opt.lr = 0.01
    opt.lr_step = 10
    opt.lr_gamma = 0.8
    # opt.attack = 'PGD-U'
    opt.summary_name = '../summary/vgg11bn_train'

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
    opt.noise = False
    report_args(opt)
    
    train(opt)

