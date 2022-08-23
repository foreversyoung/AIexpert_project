#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn


# In[2]:

class AutoEncoderConv1d(nn.Module):
    def __init__(self, channels, kernel_size):
        super(AutoEncoderConv1d, self).__init__()
        print('AutoEncoderConv1d')
        self.channels = channels
        self.kernel_size = kernel_size

        self.encoder = nn.Sequential(
            nn.Conv1d(self.channels[0], self.channels[1], kernel_size[0]),
            nn.BatchNorm1d(self.channels[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.channels[1], self.channels[2], kernel_size[1]),
            nn.BatchNorm1d(self.channels[2]),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.channels[2], self.channels[1], kernel_size[1]),
            nn.BatchNorm1d(self.channels[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(self.channels[1], self.channels[0], kernel_size[0]))

    def forward(self, x):
        f = self.encoder(x)
        output = self.decoder(f)
        return {'hidden vector': f, 'output': output}




