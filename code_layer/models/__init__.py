from .vggmodel import *
from .classifier_detector_model import *
from .assembled_detector import *
from .resnetmodel import resnet30, wide_resnet,wide_resnet_small
from .inceptionmodel import incep20
from .cwmodel import cwnet
from .detector_add_interpreter import *
from .basic_nn import BasicNN
import os
from .simclr_model import simclr_model

def load_classifier(opt):
    if opt.classifier_net == "resnet30":
        from datasets import IMAGEINDEX
        classifier = resnet30(IMAGEINDEX, pretrained=True)

    elif opt.classifier_net == "wrnet":
        model_root = opt.classifier_root + 'classifier_{}_{}_best.pth' \
            .format(opt.classifier_net, opt.dataset)

        classifier = wide_resnet(depth=28, num_classes=10, widen_factor=10) # only for cifar10
        classifier = load_GPUS(classifier, model_root) # multi-gpu trained

    else:
        model_root = opt.classifier_root + 'classifier_{}_{}_best.pth' \
            .format(opt.classifier_net, opt.dataset)
        classifier = eval(opt.classifier_net)(pretrained=False, inchannels=opt.image_channels,
                                              num_classes=opt.classifier_classes,dataset=opt.dataset)
        classifier.load_state_dict(torch.load(os.path.join(model_root), map_location=opt.device))

    print("Using classifier network: ", opt.classifier_net)
    classifier.to(opt.device)
    classifier.eval()
    return classifier

def load_detector(opt):
    detector_root = opt.detector_root + 'detector_{}_{}_{}_{}_htud({}).pth' \
        .format(opt.attack, opt.dataset, opt.detector_net, opt.interpret_method, opt.howtousedata)

    detector = eval(opt.detector_net)(False, num_classes=opt.detector_classes, inchannels=opt.map_channels,dataset=opt.dataset)
    print("Using detector htud({})".format(opt.howtousedata))

    detector.load_state_dict(torch.load(os.path.join(detector_root), map_location=opt.device))
    detector.to(opt.device)
    detector.eval()
    return detector


def load_GPUS(model, model_path, **kwargs):
    state_dict = torch.load(model_path, **kwargs)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model