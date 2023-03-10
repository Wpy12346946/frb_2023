import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
import os
import numpy as np

def seprate_dataset(savedpath, label_path , batch_size, shuffle=True, num_workers=1):
    data = []
    for one_path in savedpath:
        file_list = os.listdir(one_path)
        for file_name in file_list:
            item = np.load(os.path.join(one_path,file_name))
            shape = item.shape
            item = torch.from_numpy(item).view(1,shape[0],shape[1],shape[2])
            data.append(item)
    data = torch.cat(data)
    labels = []
    for one_path in label_path:
        label = torch.load(one_path)
        labels.append(label)
    label = torch.cat(labels)
    dataset = TensorDataset(data,label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__=='__main__':
    dataset = seprate_dataset('../../tmp_dataset/test/CW-U_9/rec_AdvInterpret','../../tmp_dataset/test/CW-U_9/CleanLabels.npy',128)
