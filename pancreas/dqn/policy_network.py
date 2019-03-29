import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    '''Our policy model'''
    model = 'model/state'

    def __init__(self, state_size, action_size, seed):
        super(PolicyNetwork, self).__init__()

        # Our neural net layers
        self.seed = torch.manual_seed(seed)
        self.l1 = nn.Linear(state_size, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_size)

    def thru_layers(self, x):
        '''
        Applies Rectified Linear Units to each layer of the
        network
        '''
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        # x = F.relu(self.l5(x))
        # x = F.relu(self.l6(x))
        # x = F.relu(self.l7(x))

        return x

    def forward(self, x):
        '''
        Forward propagates via the softmax activation function

        Params
        ------
        x - Our input feature vector
        '''
        return F.softmax(self.thru_layers(x), dim=-1)
