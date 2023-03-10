from functions import *
from models import *
import os
from datasets import build_adversarial_loader
from attack_methods import attackmethods
from interpreter_methods import interpretermethod
from tqdm import tqdm
import shutil


def interpret(opt):
    #opt.classifier_net = 'wrnet'
    clean_saliency_root = os.path.join(opt.saliency_root, '{}_{}_Clean'.format(opt.dataset, opt.classifier_net),
                                              '{}'.format(opt.interpret_method))
    if os.path.exists(clean_saliency_root):
        print("Saliency images of clean data have already exsited. \n"
              "Now to generate the saliency images of adversarial data ({})...".format(opt.attack))
        generateclean = False
    else:
        print("Now interperting clean data with method of {}...".format(opt.interpret_method))
        generateclean = True


    adver_saliency_root = os.path.join(opt.saliency_root, '{}_{}_{}'.format(opt.dataset, opt.classifier_net, opt.attack),
                                              '{}'.format(opt.interpret_method))
    if os.path.exists(adver_saliency_root):
        chars = input("\nThe saliency image file of this exp already exists "
                      "and it may be overlapped by the new ones. Continued? \n"
                      "yes: reomve the old one and generate a new one \n"
                      "no:  return\n"
                      "[yes/no]: ")
        if chars in 'yes':
            print("Removing......")
            shutil.rmtree(adver_saliency_root)
            os.makedirs(adver_saliency_root)
            print("Done!")
        if chars in 'no':
            return
        else:
            raise Exception("Wrong input!!!")

    clean_data_root = os.path.join(opt.adversarial_root, '{}_{}_Clean'.format(opt.dataset, opt.classifier_net))
    adver_data_root = os.path.join(opt.adversarial_root, '{}_{}_{}'.format(opt.dataset, opt.classifier_net, opt.attack))
    print(adver_data_root)


    # Load classifier
    #opt.classifier_net = 'vgg11bn'
    classifier = load_classifier(opt)
    if generateclean:
        dataloaders = {}
        dataloaders['train'] = build_adversarial_loader(clean_data_root, opt.train_batchsize, train=True,
                                                        num_workers=opt.workers, showname=True)
        dataloaders['test'] = build_adversarial_loader(clean_data_root, opt.val_batchsize, train=False,
                                                       num_workers=opt.workers, showname=True)
        generate_saliency_image(classifier, dataloaders, clean_saliency_root, opt)

    dataloaders = {}
    dataloaders['train'] = build_adversarial_loader(adver_data_root, opt.train_batchsize, train=True, num_workers=opt.workers, showname=True)
    dataloaders['test'] = build_adversarial_loader(adver_data_root, opt.val_batchsize, train=False, num_workers=opt.workers, showname=True)
    generate_saliency_image(classifier, dataloaders, adver_saliency_root, opt)


def generate_saliency_image(classifier, dataloaders, root, opt):
    for phase in ['test', 'train']:
        root_ = root + '/{}/'.format(phase)
        if not os.path.exists(root_):
            os.makedirs(root_)

        data_loader = dataloaders[phase]

        for image, imagename in tqdm(data_loader, desc=phase):
            image = image.to(opt.device)
            image.requires_grad = True  # Set requires_grad attribute of tensor. Important for interpret method
            b, c, w, h = image.size()

            interpreter = interpretermethod(classifier, opt.interpret_method)
            saliency_image = interpreter.interpret(image)
            interpreter.release()

            for index in range(b):
                torch.save(saliency_image[index].detach().cpu(),
                           '{}/{}{}'.format(root_, opt.interpret_method, imagename[index]))


def interpret_all(opt):
    for opt.attack in attackmethods[:-1]:
        interpret(opt)
