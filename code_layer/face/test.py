import numpy as np
import torch

from code_layer.face.nets.arcface import Arcface as arcface


def arcfacebn(pretrained=False, inchannels=3, dataset='cifar10', **kwargs):
    model = arcface(backbone='iresnet50', mode="predict")
    return model


def load_model():
    classifier_net = 'arcfacebn'
    model = eval(classifier_net)(pretrained=False, inchannels=3,
                                 num_classes=5, dataset='cifar10')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('model/arcface_iresnet50.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_batch(out, opt):
    mean_mat_feature = np.load('../data/my_file.npy', allow_pickle=True).item()
    result_batch = []
    key_ = list(mean_mat_feature.keys())
    value_ = list(mean_mat_feature.values())
    for out_now in out:
        out_now = out_now.detach().cpu().numpy()
        dis = []
        for ImgMat in value_:
            l1 = np.linalg.norm(out_now - ImgMat)
            dis.append(l1)
        sortIndex = np.argsort(dis)
        target1 = key_[sortIndex[0]]
        result_batch.append(target1)
    return torch.from_numpy(np.array(result_batch)).to(opt.device)
