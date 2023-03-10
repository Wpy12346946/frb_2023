import torch
import torch.nn as nn
import torch.nn.functional as F
from models import load_classifier
from models.assembled_detector import assembled_detector
from models.detector_add_interpreter import DetectorMultiInterpreterClassifier
from interpreter_methods import interpretermethod
from sklearn.externals import joblib


def distance_mask(grad, alpha, use_abs=False):
    b, w, h = grad.size()

    if use_abs:
        grad = grad.abs()

    grad = grad.reshape([b, -1])
    threshold = grad.min(1)[0] + alpha * (grad.max(1)[0] - grad.min(1)[0])
    threshold = threshold.reshape([b, -1])
    mask = grad.clone()

    if alpha != 1:
        mask[grad >= threshold] = 0
        mask[grad < threshold] = 1
    else:
        mask[grad < threshold+1] = 1  # alpha=1, generate original image

    mask = mask.reshape([b, w, h])
    return mask


def load_rectifier(opt, rectifier_name):
    rectifier_root = opt.rectifier_root + rectifier_name

    if 'imagenet' in opt.dataset:
        rectifier = vgg16bn(pretrained=False,  inchannels=opt.image_channels, num_classes=opt.classifier_classes)
    else:
        rectifier = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels,
                                          num_classes=opt.classifier_classes)
    print("Using rectifier {}".format(rectifier_root))

    rectifier.load_state_dict(torch.load(os.path.join(rectifier_root), map_location=opt.device))
    rectifier.to(opt.device)
    rectifier.eval()
    return rectifier


def rectify(data, ensemble, alpha, use_abs, maskvalue):
    raw_img = data[0]
    data = torch.cat(data, 1)
    b, c, w, h = data.size()
    data.detach_()
    data.requires_grad = True

    interpreter = interpretermethod(ensemble, 'VG')
    detector_grad = interpreter.interpret(data)
    interpreter.release()

    detector_grad = detector_grad.detach()
    grad = detector_grad.max(1)[0]

    mask = distance_mask(grad, alpha, use_abs)
    mask = mask.unsqueeze(1).expand(b, opt.image_channels, w, h)

    if maskvalue == -1:
        value = torch.rand_like(mask) * (mask == 0).float()
    elif maskvalue == -2:
        value = torch.normal(torch.zeros_like(mask), torch.ones_like(mask) * torch.std(raw_img)) * (
                mask == 0).float()
    else:
        value = maskvalue * (mask == 0).float()
    mask_img = (raw_img * mask + value).detach()

    return mask_img


class whole_pipline(nn.Module):
    def __init__(self,opt,alpha,interpretermethodlist=['Data', 'VG', 'IntGrad', 'GBP'],
    classes=['Clean', 'FGSM-U', 'PGD-U', 'PGD-T', 'DFool-U', 'CW-U', 'CW-T', 'DDN-U','DDN-T']):
        super(whole_pipline, self).__init__()
        detectors = []
        num = len(interpretermethodlist)
        for i in range(num):
            opt.interpret_method = interpretermethodlist[i]
            opt.map_channels = opt.image_channels
            detectorpath = opt.detector_root + 'detector_3classes_{}_{}_{}_{}set.pth' \
                .format(opt.dataset, opt.detector_net, opt.interpret_method, len(classes))
            print("Loading detector of {}".format(detectorpath))
            detector = eval(opt.detector_net)(False, num_classes=opt.detector_classes, inchannels=opt.map_channels)
            detector.load_state_dict(torch.load(os.path.join(detectorpath), map_location=opt.device))
            detector.to(opt.device)
            detector.eval()
            detectors.append(detector)

        mapchannels = [opt.image_channels] * num
        self.num=num
        self.alpha=alpha
        self.ensemble = assembled_detector(detectors=detectors, mapchannels=mapchannels)

        self.classifier = load_classifier(opt)
        print("Loading classifier")

        self.rectifier = load_rectifier(opt, opt.rectifier_name)
        print("Loading rectifier")

        self.DMIC = DetectorMultiInterpreterClassifier(classifier, detectors, interpretermethodlist, opt)

        m = "{}/{}_{}_randomforest_2nodes_{}sets.model".format(opt.tree, opt.dataset, opt.classifier_net, len(classes))
        self.rfc = joblib.load(m)


        # self.alphas = {"DDN-T": 0.95 , "PGD-T": 0.95 , "CW-T": 0.95 }


    def forward(self, data):
        data.requires_grad = True
        alphas=self.alphas
        classifier=self.classifier
        DMIC=self.DMIC
        ensemble=self.ensemble
        rectifier=self.rectifier
        rfc=self.rfc
        num=self.num
        
        classifier_out = classifier(data)

        _=DMIC(data)
        detector_output = DMIC.detector_output
        saliency_images = DMIC.saliency_images_list
        mask_img = rectify(saliency_images, ensemble, self.alpha, False, -2)
        rectifier_out = rectifier(mask_img)

        pred = [None] * num
        for i in range(num):
            pred[i] = detector_output[i].max(1)[1].unsqueeze(1)

        del saliency_images, detector_output

        preds = torch.cat(pred, 1).detach().cpu().numpy()
        forest_pred = torch.tensor(rfc.predict(preds), dtype=torch.long).to(opt.device)
        f_detect_adver = (forest_pred != 0).nonzero()[:, 0]
        f_detect_clean = (forest_pred == 0).nonzero()[:, 0]
       
        out = classifier_out.clone()
        out[f_detect_adver]=rectifier_out[f_detect_adver]

        return out
        