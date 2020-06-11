import torch.nn as nn
import torch
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units = 64, fc2_units = 64):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        noise = torch.randn_like(x)
        x =  F.tanh(noise + x)
        return x.float()



class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, h1_units=64, h2_units = 64):
        super(Critic, self).__init__()

        # Q1_FUNC
        self.fq11 = nn.Linear(state_size+action_size, h1_units)
        self.fq12 = nn.Linear(h1_units, h2_units)
        self.fq13 = nn.Linear(h2_units, 1)

        # Q2_FUNC
        self.fq21 = nn.Linear(state_size+action_size, h1_units)
        self.fq22 = nn.Linear(h1_units, h2_units)
        self.fq23 = nn.Linear(h2_units, 1)


    def forward(self, state, action):
        xs = torch.cat((state, action), dim =1)

        x1 = F.relu(self.fq11(xs))
        x1 = F.relu(self.fq12(x1))
        x1 = self.fq13(x1)

        x2 = F.relu(self.fq21(xs))
        x2 = F.relu(self.fq22(x2))
        x2 = self.fq23(x2)

        return x1, x2






