import torch
import torchvision
from torch.utils.data import DataLoader

from models.mlp import MLP

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

#
train_iterator = DataLoader(train, batch_size=128, shuffle=False)

#
test_iterator = DataLoader(test, batch_size=128, shuffle=False)

# Defines the device which should be used, e.g., `cpu` or `cuda`
device = 'cpu'

#
model = MLP(n_input=784, n_hidden=128, n_classes=10, init_weights=None, device=device)

#
model.fit(train_iterator, epochs=5)

#
model.evaluate(test_iterator)
