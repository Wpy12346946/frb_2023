import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from torchvision import transforms
from tqdm import tqdm

def gaussian(imgs, mean=0, std=0.005):
    b, c, h, w = imgs.shape
    device = imgs.device
    noise = torch.randn([b, c, h, w])*std + mean
    noise = noise.to(device)
    return imgs + noise

def smoothing(input):
    return F.max_pool2d(input, kernel_size=3, stride=1,padding=1)

def breduce(input,bits=7):
    tp = input.dtype
    input = torch.clamp(input, min=0, max=1)
    input = (input*256).type(torch.uint8)
    output = transforms.functional.posterize(input, bits)
    output = output.type(tp)
    return output/256

def scale(input):
    _,_,h,w = input.shape
    o = transforms.functional.resize(input,[int(h*1.2),int(w*1.2)])
    o = transforms.functional.center_crop(input,[h,w])
    return o

def flip(input):
    return transforms.functional.hflip(input)

def rotate(input):
    return transforms.functional.rotate(input,10)

def fft(input):
    f = torch.fft.fftshift(torch.fft.fft2(input, norm="forward"))
    def center(i,r=0.05):
        _,_,h,w = i.shape
        p = 1-r
        i[:,:,0:int(h*r),:]=0
        i[:,:,int(h*p):h,:]=0
        i[:,:,:,0:int(w*r)]=0
        i[:,:,:,int(w*p):w]=0
        return i
    fr = f.real
    fi = f.imag
    fr = center(fr)
    fi = center(fi)
    fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
    inv = torch.fft.ifft2(fft_hires, norm="forward").real
    return inv


def DKL(Z1,Z2):
    b,w = Z1.shape
    kl = F.kl_div(Z1.softmax(dim=-1).log(), Z2.softmax(dim=-1), reduction='none')
    return kl.mean(dim=-1).view(b,1)

def diff(Z1,Z2,p=2,isdiff=True):
    if isdiff:
        b,w = Z1.shape
        return torch.norm((Z1-Z2),dim=1,p=p).view(b,1)
    return torch.cat([Z1,Z2],dim=1)

class AdvJudge():
    def __init__(self,model,device,isdiff):
        self.methods = [gaussian,smoothing,breduce,scale,flip,rotate,fft]
        self.model = model
        self.device = device
        self.clf = IsolationForest(n_estimators=500,max_samples=0.8, max_features=0.5,warm_start=True)
        self.isdiff = isdiff
   
    def fit(self,dataloader):
        device = self.device
        model = self.model
        Vs = []
        for data, target in tqdm(dataloader, desc='advJudge-Fit'):
            # Send the data and label to the device
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            Zo = model(data)
            V = []
            for func in self.methods:
                imgv = func(data)
                Zv = model(imgv)
                V.append(diff(Zv,Zo,isdiff=self.isdiff))
            V = torch.cat(V,dim=1).cpu().detach()
            Vs.append(V)
        Vs = torch.cat(Vs).squeeze()
        Vs = Vs.numpy()
        self.clf.fit(Vs)
    
    def recover(self,dataloader):
        device = self.device
        model = self.model
        acc = {}
        total = 0
        for data, target in tqdm(dataloader, desc='advJudge-recover'):
            # Send the data and label to the device
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            total+=batch_size
            for func in self.methods:
                imgv = func(data)
                Zv = model(imgv)
                init_pred = Zv.max(1)[1]
                correct = torch.sum(init_pred==target.data).item()
                correct+= acc.get(func.__name__,0)
                acc[func.__name__] = correct
        for func in self.methods:
            acc[func.__name__] = acc[func.__name__]/total
        print(acc)
        return acc


    def forward(self,imgs):
        model=self.model
        device = self.device
        Zo = model(imgs)
        V = []
        for func in self.methods:
            imgv = func(imgs)
            Zv = model(imgv)
            V.append(diff(Zv,Zo,isdiff=self.isdiff))
        V = torch.cat(V,dim=1).cpu().detach().numpy()
        return self.clf.decision_function(V)

if __name__=='__main__':
    A = AdvJudge(None,None,None)
    for func in A.methods:
        print(func.__name__)
