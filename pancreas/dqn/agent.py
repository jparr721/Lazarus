import numpy as np
import random
from collections import namedtuple, deque
from policy_network import PolicyNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64  # Our minibatch size
