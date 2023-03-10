import torch
import time
import os
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def train_model(model, device, dataloaders, criterion, optimizer, scheduler,
                num_epoches=25, saved_name='', best_acc=0.0):
    since = time.time()

    best_acc = best_acc
    dataset_sizes = {'train': len(dataloaders['train'].dataset),
                     'val': len(dataloaders['val'].dataset)}

    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch + 1, num_epoches))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # save the model only when its acc is the best
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), saved_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(torch.load(saved_name))
    return model


def test_model(model, device, test_loader):
    # Accuracy and confidence counter
    correct = 0
    confd = 0
    size = len(test_loader.dataset)
    model.eval()

    # Loop over all examples in test set
    for data, target in tqdm(test_loader, desc='Test'):
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Forward pass the data through the model
        output = model(data)

        prob = torch.softmax(output, 1)
        max_prob = prob.max(1)[0]
        confd += torch.sum(max_prob).detach()

        init_pred = output.max(1)[1]  # get the index of the max log-probability
        correct += torch.sum(init_pred == target.data)

    # Calculate final accuracy and mean confidence
    final_acc = correct.double() / size
    mean_confidence = confd.double() / size
    print("Test Accuracy = {} / {} = {:.2f}%".format(correct, size, final_acc * 100))
    print("Mean Confidence   = {:.2f}%".format(mean_confidence * 100))
    return final_acc, mean_confidence


def matrix_norm(x):  # regularization
    xmin = x.min()
    xmax = x.max()
    return (x - xmin) / (xmax - xmin)


def imshow(img, cmap=None):  # show image from raw data
    img = img.detach().cpu().numpy()
    if img.ndim == 3:
        img = np.moveaxis(img, 0, 2)
    if img.ndim == 4:
        img = np.moveaxis(img, 1, 3)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if cmap is None:
        plt.imshow(img, vmin=0, vmax=1)
    else:
        plt.imshow(img, vmin=0, vmax=1, cmap=cmap)
    plt.axis("off")


# automatically choose GPU number

def check_gpus():
    '''
    GPU available check
    http://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-cuda/
    '''
    if not torch.cuda.is_available():
        print('This script could only be used to manage NVIDIA GPUs,but no GPU found in your device')
        return False
    elif not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
        print("'nvidia-smi' tool not found.")
        return False
    return True


if not check_gpus():
    raise ImportError('GPU available check failed')
else:
    def parse(line, qargs):
        '''
        line:
            a line of text
        qargs:
            query arguments
        return:
            a dict of gpu infos
        Pasing a line of csv format text returned by nvidia-smi
        解析一行nvidia-smi返回的csv格式文本
        '''
        numberic_args = ['memory.free', 'memory.total']  # 可计数的参数
        power_manage_enable = lambda v: (not 'Not Support' in v)  # lambda表达式,显卡是否滋瓷power management（笔记本可能不滋瓷）
        to_numberic = lambda v: float(v.upper().strip().replace('MIB', '').replace('W', ''))  # 带单位字符串去掉单位
        process = lambda k, v: (
            (int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
        return {k: process(k, v) for k, v in zip(qargs, line.strip().split(','))}


    def query_gpu(qargs=[]):
        '''
        qargs:
            query arguments
        return:
            a list of dict
        Querying GPUs infos
        查询GPU信息
        '''
        qargs = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit'] + qargs
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = os.popen(cmd).readlines()
        return [parse(line, qargs) for line in results]


    def by_power(d):
        '''
        helper function fo sorting gpus by power
        '''
        power_infos = (d['power.draw'], d['power.limit'])
        if any(v == 1 for v in power_infos):
            print('Power management unable for GPU {}'.format(d['index']))
            return 1
        return float(d['power.draw']) / d['power.limit']


    class GPUManager():
        '''
        qargs:
            query arguments
        A manager which can list all available GPU devices
        and sort them and choice the most free one.Unspecified
        ones pref.
        GPU设备管理器,考虑列举出所有可用GPU设备,并加以排序,自动选出
        最空闲的设备。在一个GPUManager对象内会记录每个GPU是否已被指定,
        优先选择未指定的GPU。
        '''

        def __init__(self, qargs=[]):
            '''
            '''
            self.qargs = qargs
            self.gpus = query_gpu(qargs)
            for gpu in self.gpus:
                gpu['specified'] = False
            self.gpu_num = len(self.gpus)

        def _sort_by_memory(self, gpus, by_size=False):
            if by_size:
                return sorted(gpus, key=lambda d: d['memory.free'], reverse=True)
            else:
                return sorted(gpus, key=lambda d: float(d['memory.free']) / d['memory.total'], reverse=True)

        def _sort_by_power(self, gpus):
            return sorted(gpus, key=by_power)

        def _sort_by_custom(self, gpus, key, reverse=False, qargs=[]):
            if isinstance(key, str) and (key in qargs):
                return sorted(gpus, key=lambda d: d[key], reverse=reverse)
            if isinstance(key, type(lambda a: a)):
                return sorted(gpus, key=key, reverse=reverse)
            raise ValueError(
                "The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")

        def auto_choice(self, mode=0):
            '''
            mode:
                0:(default)sorted by free memory size
            return:
                a TF device object
            Auto choice the freest GPU device,not specified
            ones
            自动选择最空闲GPU,返回索引
            '''
            for old_infos, new_infos in zip(self.gpus, query_gpu(self.qargs)):
                old_infos.update(new_infos)
            unspecified_gpus = [gpu for gpu in self.gpus if not gpu['specified']] or self.gpus

            if mode == 0:
                print('Choosing the GPU device has largest free memory...')
                chosen_gpu = self._sort_by_memory(unspecified_gpus, True)[0]
            elif mode == 1:
                print('Choosing the GPU device has highest free memory rate...')
                chosen_gpu = self._sort_by_power(unspecified_gpus)[0]
            elif mode == 2:
                print('Choosing the GPU device by power...')
                chosen_gpu = self._sort_by_power(unspecified_gpus)[0]
            else:
                print('Given an unaviliable mode,will be chosen by memory')
                chosen_gpu = self._sort_by_memory(unspecified_gpus)[0]
            chosen_gpu['specified'] = True
            index = chosen_gpu['index']
            print('Using GPU {i}'.format(i=index))
            print('memory.total:{}'.format(chosen_gpu['memory.total']))
            print('memory.free:{}'.format(chosen_gpu['memory.free']))
            return int(index)

# automatically choose GPU number
