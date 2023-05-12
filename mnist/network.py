import torch
from torch import nn

class CNNetwork(nn.Module):
  def __init__(self):
    super().__init__()

    self.flatten = nn.Flatten()
    self.softmax = nn.Softmax(dim=1)
    self.dense_layers = nn.Sequential(
      nn.Linear(28 * 28, 256),
      nn.ReLU(),
      nn.Linear(256, 10)
    )
  
  def forward(self, input):
    x = self.flatten(input)
    logits = self.dense_layers(x)
    pred = self.softmax(logits)

    return pred
