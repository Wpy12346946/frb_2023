import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models import *
from datasets import build_loader
from .utils import train_model, test_model

def classify(opt):
    '''
    Function to train an image classifier model
    :param opt:
    :param device:
    :return:
    '''

    dataloaders = {}
    dataloaders['train'] = build_loader(opt.data_root, opt.dataset, opt.train_batchsize, train=True, workers=opt.workers)
    dataloaders['val'] = build_loader(opt.data_root, opt.dataset, opt.val_batchsize, train=False, workers=opt.workers)

    # Initialize the network
    # choose classifier_net in package:models
    classifier = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    print('using network {}'.format(opt.classifier_net))
    classifier = classifier.to(opt.device)

    # Set up the weights optimizer and define the loss function
    optimizer = optim.SGD(classifier.parameters(), lr=opt.lr, momentum=opt.lr_mom, weight_decay=5e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gama)
    criterion = nn.CrossEntropyLoss()

    # Train and test the classifier model
    classifier = train_model(classifier, opt.device, dataloaders, criterion, optimizer, exp_lr_scheduler,
                saved_name= opt.classifier_root + 'classifier_{}_{}_best.pth'\
                            .format(opt.classifier_net, opt.dataset),
                num_epoches=opt.num_epoches)

    with torch.no_grad():
        test_acc, confd = test_model(classifier, opt.device, dataloaders['val'])