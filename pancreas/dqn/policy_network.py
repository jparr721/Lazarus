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
        self.l4 = nn.Linear(64, 128)
        self.l5 = nn.Linear(128, 256)
        self.l6 = nn.Linear(256, 512)
        self.l7 = nn.Linear(512, 1024)

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
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))

        return x

    def forward(self, x):
        '''
        Forward propagates via the softmax activation function

        Params
        ------
        x - Our input feature vector
        '''
        return F.softmax(self.thru_layers(x), dim=-1)
