import torch
import torch.nn as nn

class linearModule(nn.Module):
    
    def __init__(self,featurs_num = 0):
        super(linearModule,self).__init__()
        self.featurs = featurs_num
        self.layer1 = nn.Linear(self.featurs,128),
        self.layer2 = nn.Linear(128,64)
        self.layer3 = nn.Linear(64,1)

    def forword(self,x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return(x)
