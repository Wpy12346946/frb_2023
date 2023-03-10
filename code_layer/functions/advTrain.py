import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models import *
from datasets import adver_train_loader
from attack_methods import adversarialattack
import time
from tqdm import tqdm

def advtrain(opt):
    '''
    Function to train an image classifier model
    :param opt:
    :param device:
    :return:
    '''

    clean_data_root = os.path.join(opt.adversarial_root, '{}_{}_Clean'.format(opt.dataset, opt.classifier_net))
    adver_data_root = os.path.join(opt.adversarial_root, '{}_{}_{}'.format(opt.dataset, opt.classifier_net, opt.attack))

    dataloaders = {}
    dataloaders['train'] = adver_train_loader(clean_data_root,adver_data_root, opt.train_batchsize, train=True, workers=opt.workers)
    dataloaders['val'] = adver_train_loader(clean_data_root,adver_data_root, opt.val_batchsize, train=False, workers=opt.workers)

    # Initialize the network
    # choose classifier_net in package:models
    classifier = load_classifier(opt)
    classifier = classifier.to(opt.device)

    # Set up the weights optimizer and define the loss function
    optimizer = optim.SGD(classifier.parameters(), lr=opt.lr, momentum=opt.lr_mom, weight_decay=2e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gama)
    criterion = nn.CrossEntropyLoss()

    # Train and test the classifier model
    classifier = train_model(opt,classifier, opt.device, dataloaders, criterion, optimizer, exp_lr_scheduler,
                saved_name= opt.classifier_root + 'advtrainer_{}_{}_{}_best.pth'\
                            .format(opt.classifier_net, opt.dataset,opt.advtrain_alp),
                num_epoches=opt.num_epoches,alp=opt.advtrain_alp)

    # with torch.no_grad():
    test_model(opt,classifier, opt.device, dataloaders['val'])


def getAdv(data,classifier,opt):
    clean_pred = classifier(data).max(1)[1]
    targeted = opt.attack.endswith('T')
    if targeted:
        targeted_label = []
        # Randomly choose targeted label that is not the ground truth one
        for i in range(data.size(0)):
            targeted_label.append(randint(1, opt.classifier_classes-1))
        attack_label = torch.fmod(label + torch.tensor(targeted_label).long().to(device), opt.classifier_classes)
    else:
        # Untargeted attacks use the model classification labels
        attack_label = clean_pred
    eps=opt.advtrain_eps
    perturbed_data = adversarialattack(opt.attack, classifier, data, attack_label, eps, opt)
    return perturbed_data

def train_model(opt,model, device, dataloaders, criterion, optimizer, scheduler,
                num_epoches=25, saved_name='', best_acc=0.0,alp=0.5):
    since = time.time()

    best_acc = best_acc
    dataset_sizes = {'train': len(dataloaders['train'].dataset),
                     'val': len(dataloaders['val'].dataset)}

    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch+1, num_epoches))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
            #     model.train()  # Set model to training mode
            # else:
            #     model.eval()   # Set model to evaluate mode

            running_loss_clean = 0.0
            running_corrects_clean=0
            running_loss_adv=0.0
            running_corrects_adv = 0

            # Iterate over data
            for in_clean, labels in tqdm(dataloaders[phase], desc=phase):
                in_clean = in_clean.to(device)
                # in_adv=in_adv.to(device)
                in_adv=getAdv(in_clean,model,opt).detach().to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    out_clean = model(in_clean)
                    out_adv=model(in_adv)
                    _, preds_clean = torch.max(out_clean, 1)
                    _,preds_adv=torch.max(out_adv,1)
                    loss_clean = criterion(out_clean, labels)
                    loss_adv=criterion(out_adv,labels)
                    loss=alp*loss_clean+(1-alp)*loss_adv

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss_clean += loss_clean.item() * in_clean.size(0)
                running_loss_adv += loss_adv.item() * in_adv.size(0)
                running_corrects_clean += torch.sum(preds_clean == labels.data)
                running_corrects_adv += torch.sum(preds_adv == labels.data)

            epoch_loss_clean = running_loss_clean / dataset_sizes[phase]
            epoch_loss_adv = running_loss_adv / dataset_sizes[phase]
            epoch_acc_clean = running_corrects_clean.double() / dataset_sizes[phase]
            epoch_acc_adv = running_corrects_adv.double() / dataset_sizes[phase]

            print('{} Loss_clean: {:.4f} Loss_adv: {:.4f} Acc_clean: {:.4f} Acc_adv:{:.4f}'.format(
                phase, epoch_loss_clean,epoch_loss_adv, epoch_acc_clean,epoch_acc_adv))

            # save the model only when its acc is the best
            if phase == 'val' and epoch_acc_adv > best_acc:
                best_acc = epoch_acc_adv
                torch.save(model.state_dict(), saved_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(torch.load(saved_name) )
    return model

def test_model(opt,model, device, test_loader):
    # Accuracy and confidence counter
    correct_clean = 0
    correct_adv=0
    confd_clean = 0
    confd_adv=0
    size = len(test_loader.dataset)
    # model.eval()

    # Loop over all examples in test set
    for data_clean, target in tqdm(test_loader, desc='Test'):
        # Send the data and label to the device
        data_clean, target = data_clean.to(device), target.to(device)
        data_adv=getAdv(data_clean,model,opt).detach().to(device)
        # Forward pass the data through the model
        out_clean = model(data_clean)
        prob = torch.softmax(out_clean, 1)
        max_prob = prob.max(1)[0]
        confd_clean += torch.sum(max_prob).detach()

        out_adv = model(data_adv)
        prob = torch.softmax(out_adv, 1)
        max_prob = prob.max(1)[0]
        confd_adv += torch.sum(max_prob).detach()

        init_pred = out_clean.max(1)[1] # get the index of the max log-probability
        correct_clean += torch.sum(init_pred == target.data)

        init_pred = out_adv.max(1)[1] # get the index of the max log-probability
        correct_adv += torch.sum(init_pred == target.data)

    # Calculate final accuracy and mean confidence
    final_acc_clean = correct_clean.double() / size
    mean_confidence_clean = confd_clean.double() / size
    final_acc_adv = correct_adv.double() / size
    mean_confidence_adv = confd_adv.double() / size
    print("Test Accuracy Clean = {} / {} = {:.2f}%".format(correct_clean, size, final_acc_clean * 100))
    print("Mean Confidence Clean   = {:.2f}%".format(mean_confidence_clean * 100))
    print("Test Accuracy Adv = {} / {} = {:.2f}%".format(correct_adv, size, final_acc_adv * 100))
    print("Mean Confidence Adv  = {:.2f}%".format(mean_confidence_adv * 100))
