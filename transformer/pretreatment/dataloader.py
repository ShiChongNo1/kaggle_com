from enum import Enum

import torch
import numpy as np
import pandas as pd
from joblib import dump
from pretreatment.datamachine import scale
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class DataloaderType(Enum):
    train = 0
    validate = 1
    test = 2
    prediction = 3


def data_load(config: EasyDict, dataType: DataloaderType):

    if dataType == DataloaderType.prediction:
        dataset = pd.read_csv(config.load.test_path)
        dataset.insert(loc=dataset.shape[1]+1, column='target', value=0)
    else:
        dataset = pd.read_csv(config.load.train_path)
        trian_len = int(len(dataset)*config.train.train_scale)
        val_len = int(len(dataset)*config.train.val_scale)
        if dataType == DataloaderType.train:
            dataset = dataset.iloc[:trian_len, :]
        elif dataType == DataloaderType.validate:
            dataset = dataset.iloc[val_len:, :]
        elif dataType == DataloaderType.test:
            dataset = dataset.iloc[trian_len + val_len:, :]
            pass

    return dataset


class StoreDataset(Dataset):
    def __init__(self, config: EasyDict, data_type: DataloaderType) -> None:
        super(StoreDataset, self).__init__()
        self.dl = data_load(config, data_type)
        self.transform = MinMaxScaler()
        self.dl_input = self.dl.iloc[:, :-1].values
        self.dl_target = self.dl.iloc[:, [-1]].values
        self.length = config.train.training_length

    def __len__(self):
        return int(len(self.dl)/self.length) - 2
        # return len(self.dl)

    def __getitem__(self, index):
        _input = torch.tensor(
            self.dl_input[index*self.length: (index+1)*self.length])
        _input = torch.where(torch.isinf(
            _input), torch.full_like(_input, 0), _input)
        _input = torch.where(torch.isnan(
            _input), torch.full_like(_input, 0), _input)

        target = torch.tensor(
            self.dl_target[index * self.length: (index + 1) * self.length])
        return scale(_input), scale(target)
