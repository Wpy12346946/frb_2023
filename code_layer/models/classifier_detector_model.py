import torch
import torch.nn as nn
import torch.nn.functional as F

from interpreter_methods import interpretermethod

class C_D(nn.Module):
    def __init__(self, classifier, detector, opt):
        super(C_D, self).__init__()
        self.classifier = classifier
        self.detector = detector
        self.opt = opt
        self.classifier_output = None

    def forward(self, x):
        x = x.clone()
        if x.requires_grad and not x.is_leaf:
            x.retain_grad()

        if not x.requires_grad:
            x.requires_grad = True

        unknown = interpreter(self.classifier, self.opt.interpret_method)
        # saliency_maps = unknown.get_saliency_maps(x)
        saliency_maps = unknown.get_datagrad(x)
        saliency_maps = saliency_maps[:,self.opt.map_channels_start:self.opt.map_channels_start+self.opt.map_channels,::]
        self.classifier_output = unknown.model_output
        unknown.release()
        del unknown

        self.det_output = self.detector(saliency_maps)# value > 0, is adversarial
        self.is_adversarial = self.det_output[:, 0] - self.det_output[:, 1] # value > 0, is adversarial

        maxclassifier = torch.max(self.classifier_output, 1)[0]
        N_1 = (self.is_adversarial+1)*maxclassifier
        return torch.cat([self.classifier_output, N_1.unsqueeze(1)], 1)

class C_2D(nn.Module):
    def __init__(self, classifier, detector1, detector2,opt):
        super(C_2D, self).__init__()
        self.classifier = classifier
        self.detector1 = detector1
        self.detector2 = detector2
        self.opt = opt
        self.classifier_output = None


    def forward(self, x):
        x = x.clone()
        if x.requires_grad and not x.is_leaf:
            x.retain_grad()

        if not x.requires_grad:
            x.requires_grad = True

        unknown = interpreter(self.classifier, self.opt.interpret_method)
        # saliency_maps = unknown.get_saliency_maps(x)
        saliency_maps = unknown.get_datagrad(x)
        saliency_maps = saliency_maps[:,self.opt.map_channels_start:self.opt.map_channels_start+self.opt.map_channels,::]
        self.classifier_output = unknown.model_output
        unknown.release()
        del unknown

        self.det_output1 = self.detector1(saliency_maps[:,:self.opt.map_channels//2,::])
        self.det_output2 = self.detector2(saliency_maps[:, self.opt.map_channels//2:, ::])

        a =  self.det_output1[:,0] - self.det_output1[:,1]
        b = self.det_output2[:,0] - self.det_output2[:,1]
        output = a * (a.abs() > b.abs()).float() + b * (a.abs() <= b.abs()).float()
        self.is_adversarial = output


        # a = self.det_output1
        # b = self.det_output2
        # self.is_adverarial = a*(a>b).float()+b*(a<=b).float()

        # print("C2D, ab", a[:10], b[:10])
        # print("C2D", self.is_adverarial[:10])

        maxclassifier = torch.max(self.classifier_output, 1)[0]
        N_1 = (self.is_adversarial+1)*maxclassifier
        return torch.cat([self.classifier_output, N_1.unsqueeze(1)], 1)

class C_3D(nn.Module):
    def __init__(self, classifier, detector1, detector2, detector3, opt):
        super(C_3D, self).__init__()
        self.classifier = classifier
        self.detector1 = detector1
        self.detector2 = detector2
        self.detector3 = detector3
        self.opt = opt
        self.classifier_output = None


    def forward(self, x):
        x = x.clone()
        if x.requires_grad and not x.is_leaf:
            x.retain_grad()

        if not x.requires_grad:
            x.requires_grad = True

        unknown = interpreter(self.classifier, self.opt.interpret_method)
        # saliency_maps = unknown.get_saliency_maps(x)
        saliency_maps = unknown.get_datagrad(x)
        saliency_maps = saliency_maps[:,self.opt.map_channels_start:self.opt.map_channels_start+self.opt.map_channels,::]
        self.classifier_output = unknown.model_output
        unknown.release()
        del unknown

        self.det_output1 = self.detector1(saliency_maps[:,:self.opt.map_channels//2,::])[:,0]
        self.det_output2 = self.detector2(saliency_maps[:, self.opt.map_channels//2:, ::])[:,0]
        self.det_output3 = self.detector3(saliency_maps)[:,0]

        a = self.det_output1
        b = self.det_output2
        c = self.det_output3

        self.is_adverarial = a*((a>b)*(a>c)).float()+b*((b>=a)*(b>c)).float()+c*((c>=a)*(c>=b)).float()


        maxclassifier = torch.max(self.classifier_output, 1)[0]
        N_1 = (self.is_adverarial+1)*maxclassifier
        return torch.cat([self.classifier_output, N_1.unsqueeze(1)], 1)