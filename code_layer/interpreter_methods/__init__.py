from .GuidedBackprop import GuidedBackprop
from .VanilaGradients import VanilaGradients
from .GradCAM import GradCAM
from .LRP import LRP
from .IntegratedGrad import IntegratedGrad
from .SmoothGrad import SmoothGrad
from .DeepLIFT import DL
from .LayerWise_VG import LayerWise_VG

def interpretermethod(model, whichmethod, specific_layer=None,**kwargs):
    # Choose interpret method to generate interpreting maps
    if whichmethod == 'VG':
        interpreter = VanilaGradients(model)
    elif whichmethod == 'GBP':
        interpreter = GuidedBackprop(model)
    elif whichmethod == 'GCAM':
        interpreter = GradCAM(model, specific_layer=specific_layer)
    elif whichmethod == "LRP":
        interpreter = LRP(model)
    elif whichmethod == 'IG':
        interpreter = IntegratedGrad(model)
    elif whichmethod == 'SmGrad':
        interpreter = SmoothGrad(model)
    elif whichmethod == 'DL' or whichmethod == 'DeepLIFT':
        interpreter = DL(model)
    elif whichmethod == 'LVG' or whichmethod == 'LayerWise_VG':
        interpreter = LayerWise_VG(model, **kwargs)
    else:
        raise Exception('No such a saliency method ({})'.format(whichmethod))
    return interpreter
