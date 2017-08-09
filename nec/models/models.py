import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class AtariDQN(nn.Module):
  def __init__(self, embedding_size):
    super(AtariDQN, self).__init__()

    self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    self.fc = nn.Linear(3136, 512)
    self.head = nn.Linear(512, embedding_size)

    self.initialize_weights()

  def initialize_weights(self):
    conv_layers = [v for k,v in self._modules.iteritems() if 'conv' in k]
    for layer in conv_layers:
      init.xavier_uniform(layer.weight)
    init.xavier_uniform(self.head.weight)
    init.xavier_uniform(self.fc.weight)

  def forward(self, x):
    out = F.selu((self.conv1(x)))
    out = F.selu(self.conv2(out))
    out = F.selu(self.conv3(out))
    out = F.selu(self.fc(out.view(out.size(0), -1)))
    out = self.head(out)
    return out

class CartPoleDQN(nn.Module):
  def __init__(self, embedding_size):
    super(CartPoleDQN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.head = nn.Linear(448, embedding_size)

  def initialize_weights(self):
    conv_layers = [v for k,v in self._modules.iteritems() if 'conv' in k]
    for layer in conv_layers:
      init.xavier_uniform(layer.weight)
    init.xavier_uniform(self.head.weight)

  def forward(self, x):
    x = F.selu(self.conv1(x))
    x = F.selu(self.conv2(x))
    x = F.selu(self.conv3(x))
    return self.head(x.view(x.size(0), -1))
