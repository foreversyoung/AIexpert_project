import torch
import torch.nn as nn
import torch.nn.functional as F

_flatten = lambda l: [item for sublist in l for item in sublist]

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

        self.encode_params = _flatten([container.parameters() for container in [self.encoder])
        self.decode_params = _flatten([container.parameters() for container in [self.decoder]])

    def forward(self, x):
        x = self.encoder(x)
        output = self.decoder(f)

        return x

    def set_train_encoder(self):
        self.mode = 'TRAIN_ENCODER'
        for param in self.encode_params:
            param.requires_grad = True
        return self.encode_params

    def set_train_decoder(self):
        self.mode = 'TRAIN_DECODER'
        for param in self.encode_params:
            param.requires_grad = False
        return self.decode_params

    def set_eval_encoder(self):
        self.mode = 'EVAL_ENCODER'
        return []

    def set_eval_decoder(self):
        self.mode = 'EVAL_DECODER'
        return []