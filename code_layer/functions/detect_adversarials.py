import torch
import torch.nn as nn
from models import *
import torch.optim as optim
from torch.optim import lr_scheduler
from datasets import build_maps_loader
from .utils import train_model, test_model
import os


def detect(opt, howtousedata=None):
    if howtousedata is None:
        howtousedata = opt.howtousedata

    adversarial_root = opt.adversarial_root + '{}_{}_{}/'.format(opt.dataset, opt.classifier_net, opt.attack)
    saliency_root = opt.saliency_root + '{}_{}_{}/{}/'.format(opt.dataset, opt.classifier_net, opt.attack, opt.interpret_method)
    train_loader, map_channels = build_maps_loader(adversarial_root, saliency_root, opt.interpret_method,train=True,
                   howtousedata=howtousedata, batch_size=opt.train_batchsize, num_workers=opt.workers)
    val_loader, map_channels = build_maps_loader(adversarial_root, saliency_root, opt.interpret_method,train=False,
                   howtousedata=howtousedata, batch_size=opt.val_batchsize, num_workers=opt.workers)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Initialize the network
    if opt.detector_mode=='train':
        print("Training detector")
        detectorpath = opt.detector_root + 'detector_{}_{}_{}_{}_htud({}).pth' \
            .format(opt.attack, opt.dataset, opt.detector_net, opt.interpret_method, howtousedata)
        opt.map_channels = map_channels
        detector = eval(opt.detector_net)(False, num_classes=opt.detector_classes, inchannels=opt.map_channels)
        detector = detector.to(opt.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(detector.parameters(), lr=opt.lr, momentum=opt.lr_mom)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gama)

        if os.path.exists(detectorpath):
            chars = input(
                "\nThe training detector has already existed. Continued? \n"
                "yes: reomve the old one and generate a new one \n"
                "no:  break the program\n"
                "[yes/no]: ")
            if chars in 'yes':
                print("Removing......")
                os.remove(detectorpath)
                print("Done!")
            elif chars in 'no':
                raise Exception("The detector has already existed!!!")

        detector = train_model(detector, opt.device, dataloaders, criterion, optimizer, exp_lr_scheduler,
                            saved_name=detectorpath, num_epoches=opt.num_epoches, best_acc=0.0)
        print("{} Detector htud{} of {} under {} attack trained successfully."\
              .format(opt.interpret_method, opt.howtousedata, opt.dataset, opt.attack))
    else:
        detector=load_detector(opt)

    print("Testing detector")
    with torch.no_grad():
        test_acc, confd = test_model(detector, opt.device, val_loader)
    return

def detect_all(opt):

    # only data
    detect(opt,  howtousedata=0)

    # only saliency
    detect(opt,  howtousedata=1)

    # # torch.cat([data,saliency])
    # detect(opt,  howtousedata=2)
