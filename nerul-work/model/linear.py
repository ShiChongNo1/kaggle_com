import torch
import torch.nn as nn


class linearModule(nn.Module):

    def __init__(self, input_num=1):
        super(linearModule, self).__init__()
        self.featurs = input_num
        self.layers = nn.Sequential(
            nn.Linear(self.featurs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.layers(x)
