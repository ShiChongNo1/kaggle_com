import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class eloDataset(Dataset):
    def __init__(self, data):
        self.input_num = data.shape[0]
        self.features = torch.Tensor(data[:,:-1])
        self.target = torch.Tensor(data[:,[-1]])
        self.features = torch.where(torch.isinf(self.features), torch.full_like(self.features, 0), self.features)
    
    def __getitem__(self,index):
        return self.features[index], self.target[index]
    
    def __len__(self):
        return self.input_num

def loader_database(data):

    train_dataset = data[:int(len(data)*0.85)]
    val_dataset = data[int(len(data)*0.85):]

    train_sample = eloDataset(train_dataset)
    val_sample = eloDataset(val_dataset)

    train_loader = DataLoader(dataset=train_sample, batch_size=32, shuffle=True, num_workers=9)
    val_loader = DataLoader(dataset=val_sample, batch_size=32, shuffle=True, num_workers=9)

    return train_loader, val_loader




