import torch.nn as nn
import torch.nn.functional as F

_flatten = lambda l: [item for sublist in l for item in sublist]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # input: b, 1, 28, 28
        # output: b, 8, 2, 2
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        # output: b, 10
        self.classifier = nn.Sequential(
            nn.Linear(8 * 2 * 2, 10)
        )

        # output: b, 8*2*2
        self.unclassifier = nn.Sequential(
            nn.Linear(10, 8 * 2 * 2)
        )
        # output: b, 1, 28, 28
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

        self.encode_params = _flatten([container.parameters() for container in [self.encoder, self.classifier]])
        self.decode_params = _flatten([container.parameters() for container in [self.unclassifier, self.decoder]])

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 8 * 2 * 2)
        x = self.classifier(x)

        if self.mode in ['TRAIN_ENCODER', 'EVAL_ENCODER']:
            x = F.log_softmax(x, dim=1)
        else:
            x = self.unclassifier(x)
            x = x.view(-1, 8, 2, 2)
            x = self.decoder(x)

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
