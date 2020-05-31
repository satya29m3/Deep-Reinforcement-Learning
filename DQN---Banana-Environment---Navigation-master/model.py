from torch import nn
import torch
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self,action_size,state_size,seed):
        super(Net,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_size,24)
        self.linear2 = nn.Linear(24,12)
        self.linear3 = nn.Linear(12,4)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

