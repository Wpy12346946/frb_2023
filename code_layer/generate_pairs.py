import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
from models.assembled_detector import assembled_detector
from interpreter_methods import interpretermethod
from Options import Options
# from sklearn.externals import joblib
from datasets import *
from datasets import pairs_loader
from tqdm import tqdm
import torchvision.utils as vutils

def main():
    device = opt.device
    loader = {}
    loader['test'] = build_loader(opt.data_root, opt.dataset, opt.val_batchsize, train=False, workers=0)
    loader['train'] = build_loader(opt.data_root, opt.dataset, opt.val_batchsize, train=True, workers=0)
    saved_name = f'../classifier_pth/classifier_{opt.classifier_net}_{opt.dataset}_best.pth'
    model = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels, num_classes=opt.classifier_classes,dataset=opt.dataset)
    print('using network {}'.format(opt.classifier_net))
    print('loading from {}'.format(saved_name))
    model.load_state_dict(torch.load(saved_name,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()
    classifier=model

    opt.tmp_dataset = f'tmp_dataset_{opt.classifier_net}_{opt.dataset}'
    for phase in ['test','train']:
        pairs=[]
        cnt = 0
        for data, label in tqdm(loader[phase], desc='choose_images'):
            data,label=data.to(device),label.to(device)  
            out=classifier(data).max(1)[1]
            if(torch.sum(out==label).item()==0):
                continue
            cnt+=1
            if cnt>=opt.max_cnt:
                break
            num_samples = 50
            samples = set(random.sample(range(len(loader[phase])), num_samples))
            for i, (xi, yi) in enumerate(loader[phase]):
                xi,yi=xi.to(device),yi.to(device)
                if i not in samples:
                    continue
                out=classifier(xi).max(1)[1]
                if(torch.sum(out==yi).item()==0):
                    continue
                if(torch.sum(yi==label).item()==0):
                    # print(label[0],yi[0])
                    Org = data[0].clone().detach().cpu()
                    Org_label = label[0].clone().detach().cpu()
                    Pair = xi[0].clone().detach().cpu()
                    Pair_label = yi[0].clone().detach().cpu()
                    pairs.append({'Org':Org,'Org_label':Org_label,'Pair':Pair,'Pair_label':Pair_label})
                    break
        torch.save(pairs,f'../data/{opt.dataset}/{phase}_pair.pth')

        root = f'../data/{opt.dataset}/{phase}_pair.pth'
        pairs = pairs_loader(root,16)
        for org,org_l,pair,pair_l in pairs:
            vutils.save_image(org,f'../{opt.tmp_dataset}/test_org.png')
            vutils.save_image(pair,f'../{opt.tmp_dataset}/test_pair.png')
            print(org_l)
            print(pair_l)
            print(torch.norm(org-pair,p=4))
            break

if __name__=='__main__':
    opt = Options().parse_arguments()
    opt.device=torch.device("cpu")
    print(opt.device)
    opt.max_cnt = 10000
    for opt.dataset in ['imagenet']:
        # opt.dataset = 'fmnist'
        # opt.dataset = 'imagenet'
        if 'mnist' in opt.dataset:
            opt.image_channels = 1
            opt.train_batchsize = opt.val_batchsize = 1
            opt.classifier_net = 'vgg11bn'
            opt.classifier_classes=10

        if 'cifar' in opt.dataset:
            opt.train_batchsize = opt.val_batchsize = 1
            opt.classifier_net = 'vgg11bn'
            opt.classifier_classes=10

        if 'image' in opt.dataset:
            opt.train_batchsize = opt.val_batchsize = 1
            opt.classifier_net = 'wide_resnet'
            opt.classifier_classes=30
            opt.max_cnt = 500

        main()