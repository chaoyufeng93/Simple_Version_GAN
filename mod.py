import torch
import os
from glob import glob
import random
import math
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F

class LinearBLK(torch.nn.Module):
  def __init__(self, input_dim, output_dim):
    super(LinearBLK, self).__init__()
    self.linear = torch.nn.Linear(input_dim, output_dim)
    self.norm = torch.nn.BatchNorm1d(output_dim)
    self.act = torch.nn.LeakyReLU()

  def forward(self, x):
    x = self.act(self.norm(self.linear(x)))

    return x
   
class GAN_Generator(torch.nn.Module):
  def __init__(self, input_dim):
    super(GAN_Generator, self).__init__()
    self.in_linear = torch.nn.Linear(input_dim, 784)
    self.norm_in = torch.nn.BatchNorm1d(784)
    self.act_in = torch.nn.LeakyReLU()
    self.act_out = torch.nn.Tanh()

    self.connect = torch.nn.ModuleList([
        LinearBLK(784, 784*2),
        LinearBLK(784*2, 784*3),
        LinearBLK(784*3, 784*2),
        LinearBLK(784*2, 784),
    ])


  def forward(self, x):
    x = self.act_in(self.norm_in(self.in_linear(x)))
    for i in range(4):
      x = self.connect[i](x)
    x = self.act_out(x)
    x = x.reshape(x.shape[0],1,28,28)

    return x
  
# input: (batch_size, 1, 28, 28)
class GAN_Discrim(torch.nn.Module):
  def __init__(self, input_dim, linear_dim):
    super(GAN_Discrim, self).__init__()
    self.linear_in = torch.nn.Linear(input_dim, linear_dim)
    self.linear2 = torch.nn.Linear(linear_dim, 3*linear_dim)
    self.linear3 = torch.nn.Linear(3*linear_dim, 2*linear_dim)
    self.linear_out = torch.nn.Linear(2*linear_dim, 1)
    self.act = torch.nn.LeakyReLU()
    self.sig = torch.nn.Sigmoid()

  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.act(self.linear_in(x))
    x = self.act(self.linear2(x))
    x = self.act(self.linear3(x))
    x = self.linear_out(x)
    x = self.sig(x)
    return x.view(x.shape[0], -1)
