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
        self.l3 = nn.Linear(64, 64)

    def get_state_value(self, x):
        x = self.thru_layers(x)
        return self.value_head(x)

    def load_state(self):
        self.load_state_dict(torch.load(PolicyNetwork.model_path))

    def save_state(self):
        torch.save(self.state_dict(), PolicyNetwork.model_path)

    def thru_layers(self, x):
        '''
        Applies Rectified Linear Units to each layer of the
        network
        '''
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x

    def forward(self, x):
        '''
        Forward propagates via the softmax activation function

        Params
        ------
        x - Our input feature vector
        '''
        # x = self.thru_layers(x)
        # action_scores = self.action_head(x)
        # return F.softmax(action_scores, dim=-1)
        return self.thru_layers(x)
