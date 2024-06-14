from torchvision import datasets, transforms
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.relu = nn.ReLU()
    self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
    self.flatten = nn.Flatten()
    self.lin1 = nn.Linear(16 * 28 * 28, 128)
  

  def forward(self, x):
    x1 = self.relu(self.conv1(x))
    x2 = self.flatten(x1)
    x3 = self.relu(self.lin1(x2))
    return x3

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.relu = nn.ReLU()
    self.lin1 = nn.Linear(128, 16 * 28 * 28)
    self.unflatten = nn.Unflatten(dim=1, unflattened_size=(16, 28, 28))
    self.deconv1 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)
    self.sig = nn.Sigmoid()

  def forward(self, x):
    x1 = self.relu(self.lin1(x))
    x2 = self.unflatten(x1)
    x3 = self.sig(self.deconv1(x2))
    return x3

class AutoEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, x):
    enc = self.encoder(x)
    dec = self.decoder(enc)

    return dec