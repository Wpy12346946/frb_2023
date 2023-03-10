import torch
import torch.nn as nn
import torch.nn.functional as F

from interpreter_methods import interpretermethod


class DetectorInterpreter(nn.Module):
    def __init__(self, classifier, detector, opt):
        super(DetectorInterpreter, self).__init__()
        self.classifier = classifier
        self.detector = detector
        self.opt = opt
        self.classifier_output = None

    def forward(self, x):
        if x.requires_grad and not x.is_leaf:
            x.retain_grad()

        if not x.requires_grad:
            x.requires_grad = True

        if self.opt.interpret_method != 'Data':
            interpreter = interpretermethod(self.classifier, self.opt.interpret_method)
            saliency_images = interpreter.interpret(x)
            #self.classifier_output = interpreter.model_output
            self.classifier_output = self.classifier(x)
            interpreter.release()
            del interpreter
        else:
            self.classifier_output = self.classifier(x)
            saliency_images = x

        self.detector_output = self.detector(saliency_images)

        return self.detector_output


class DetectorInterpreterClassifier(nn.Module):
    def __init__(self, classifier, detector, opt):
        super(DetectorInterpreterClassifier, self).__init__()
        self.classifier = classifier
        self.detector = detector
        self.opt = opt
        self.classifier_output = None

    def forward(self, x):
        if x.requires_grad and not x.is_leaf:
            x.retain_grad()

        if not x.requires_grad:
            x.requires_grad = True

        if self.opt.interpret_method != 'Data':
            interpreter = interpretermethod(self.classifier, self.opt.interpret_method)
            saliency_images = interpreter.interpret(x)
            self.classifier_output = self.classifier(x)
            interpreter.release()
            del interpreter
        else:
            self.classifier_output = self.classifier(x)
            saliency_images = x

        self.detector_output = self.detector(saliency_images)

        detector_prob = F.softmax(self.detector_output, 1)
        self.is_adversarial = 1/3 - detector_prob[:, 0]  # value > 0, is adversarial

        #detector_pred = self.detector_output.max(1)[1]
        #self.is_adversarial =  detector_pred.float() - 0.5

        maxclassifier = torch.max(self.classifier_output, 1)[0]

        N_1 = (self.is_adversarial + 1) * maxclassifier
        return torch.cat([self.classifier_output, N_1.unsqueeze(1)], 1)


class DetectorMultiInterpreterClassifier(nn.Module):
    def __init__(self, classifier, detectors, interpretmethods, opt):
        super(DetectorMultiInterpreterClassifier, self).__init__()
        self.classifier = classifier
        self.detectors = detectors
        self.opt = opt
        self.interpretmethods = interpretmethods
        self.classifier_output = None

    def forward(self, x):
        if x.requires_grad and not x.is_leaf:
            x.retain_grad()

        if not x.requires_grad:
            x.requires_grad = True

        assert len(self.detectors) == len(self.interpretmethods)
        num = len(self.detectors)

        self.saliency_images_list = []
        self.detector_output = []
        self.is_adversarial = []
        for i, self.opt.interpret_method in enumerate(self.interpretmethods):
            if self.opt.interpret_method != 'Data':
                interpreter = interpretermethod(self.classifier, self.opt.interpret_method)
                saliency_images = interpreter.interpret(x)
                self.classifier_output = self.classifier(x)
                interpreter.release()
                del interpreter
            else:
                self.classifier_output = self.classifier(x)
                saliency_images = x

            self.saliency_images_list.append(saliency_images)
            output = self.detectors[i](saliency_images)
            self.detector_output.append(output)
            detector_prob = F.softmax(output, 1)
            self.is_adversarial.append(1/3 - detector_prob[:, 0])


        N_ = [None] * num
        maxclassifier = torch.max(self.classifier_output, 1)[0]

        for i in range(num):
            N_[i] = (self.is_adversarial[i] + 1) * maxclassifier
            N_[i] = N_[i].unsqueeze(1)

        return torch.cat([self.classifier_output]+N_, 1)