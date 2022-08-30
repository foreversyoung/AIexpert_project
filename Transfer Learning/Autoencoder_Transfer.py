# MNIST autoencoder with transfer learning
## Example

## PyTorch + MNIST -> Transfer Learning

## Source Code: autoencoder_SPS.py
## PyTorch Training: trainer.py

# Load data

import torch
import torchvision
from torchvision import transforms
import digits_dataset

transform = torchvision.transforms.Compose(
    [
    transforms.Resize((28, 28)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# The standard MNIST dataset with handwritten digits plus labels.
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=5,
                                          shuffle=True)

# The digit training set is used to train the encoder portion of the network.
label_transform = lambda l: l  # We'll use the label_transform further down.
digit_trainset = digits_dataset.MnistWithPrintWriterLabels(trainset,
                                            "./data/digits",
                                            digit_transform=transform,
                                                          label_transform=lambda l: label_transform(l))
digit_trainset_loader = torch.utils.data.DataLoader(digit_trainset, batch_size=5,
                                                    shuffle=True)

# Test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)


# Show some sample MNIST digits, corresponding labels, and target font digits.
from util import imshow

images, labels = iter(train_loader).next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % labels[j] for j in range(5)))

images, digits = iter(digit_trainset_loader).next()
imshow(torchvision.utils.make_grid(images))
imshow(torchvision.utils.make_grid(digits))

# Define network
from autoencoder import Net
net = Net()
print(net)

# Train encoder
import torch.optim as optim
import torch.nn as nn
from trainer import train

# The Net::set_train_encoder function returns the set of parameters
# in the encoder portion of the network. We pass those parameters to the
# optimizer
params = net.set_train_encoder()
optimizer = optim.SGD(params, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
train(net, train_loader, optimizer, criterion, 1)


# Train decoder
# Important: Only pass those parameters to the optimizer that should get updated.
# The optimizer will change weights unless the gradient is 'None', i.e., the
# interaction between the optimizer and require_grad is not so obvious. Alternatively,
# we could set the encoder weights to None explicitly in Net::set_train_decoder,
# but that seems more of a hack.
# Compare https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py#L62
params = net.set_train_decoder()
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()
train(net, digit_trainset_loader, optimizer, criterion, 1)


# Test network
images, labels = iter(train_loader).next()

# Display input test images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(len(images))))

# Test encoder portion of the network
net.set_eval_encoder()
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted by encoder portion of network: ', ' '.join('%5s' % predicted[j] for j in range(len(images))))

# Test decoder portion of the network
net.set_eval_decoder()
outputs = net(images)
imshow(torchvision.utils.make_grid(outputs))


# Determine the number of parameters of the decoder network
params = net.set_train_decoder()
print("Number of float weights in decoder network:", sum(p.numel() for p in net.parameters()))

# Retrain the decoder to render ((input + 1) mod 10).
label_transform = lambda l: (int(l) + 1) % 10
params = net.set_train_decoder()
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()
train(net, digit_trainset_loader, optimizer, criterion, 1)

# Visualize re-trained decoder
images, labels = iter(test_loader).next()
imshow(torchvision.utils.make_grid(images))
net.set_eval_decoder()
outputs = net(images)
imshow(torchvision.utils.make_grid(outputs))