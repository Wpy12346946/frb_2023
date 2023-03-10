import torch
import torch.nn as nn
import torch.nn.functional as F


class assembled_detector(nn.Module):
    def __init__(self, detectors, mapchannels):
        super(assembled_detector, self).__init__()
        self.detectors = detectors
        self.mapchannels = mapchannels
        assert len(self.mapchannels) == len(self.detectors)

    def forward(self, detector_inputs):
        detector_outputs = []
        entropys = []

        channel_pointer = 0
        for i in range(len(self.detectors)):
            detector = self.detectors[i]
            output = detector(detector_inputs[:,channel_pointer:channel_pointer+self.mapchannels[i],:,:])
            channel_pointer += self.mapchannels[i]
            detector_outputs.append(output.unsqueeze(1))
            # detector_outputs.append(output)
            p = F.softmax(output, 1)
            entropy = -torch.sum(p * torch.log(p), 1)
            entropys.append(entropy.unsqueeze(1))
            # entropys.append(entropy)

        entropys = torch.cat(entropys, 1)
        detector_outputs = torch.cat(detector_outputs, 1)

        min_args = torch.argmin(entropys, dim=1)
        onehot = torch.zeros_like(entropys)
        onehot = onehot.scatter_(1, min_args.unsqueeze(1), 1).unsqueeze(2)
        onehot = torch.cat([onehot, onehot, onehot], dim=2)
        output = torch.sum(onehot * detector_outputs, dim=1)

        # num=3
        # if num == 2:
        #     a, b = entropys[0], entropys[1]
        #     output = detector_outputs[0] * (a<b).unsqueeze(-1).float() + detector_outputs[1] * (a >= b).unsqueeze(-1).float()
        # elif num ==3:
        #     a, b, c = entropys[0], entropys[1], entropys[2]
        #     print('a',a.requires_grad)
        #     print('a',a.is_leaf)
        #     output = detector_outputs[0]*(a<b).unsqueeze(-1).float()*(a<c).unsqueeze(-1).float() \
        #              + detector_outputs[1]*(b<=a).unsqueeze(-1).float()*(b<c).unsqueeze(-1).float() \
        #              + detector_outputs[2]*(c<=b).unsqueeze(-1).float()*(c<=a).unsqueeze(-1).float()
        # else:
        #     raise Exception("Assemble detectors error")
        return output
