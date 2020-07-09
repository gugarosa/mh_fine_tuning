import torch
from torch import nn
from torch.nn import functional as F

from core.model import Model


class MLP(Model):
    """An MLP class implements a Multi-Layer Perceptron learning architecture.

    """

    def __init__(self, n_input=784, n_hidden=128, n_classes=10, lr=0.001, init_weights=None, device='cpu'):
        """Initialization method.

        Args:
            n_input (int): Number of input units.
            n_hidden (int): Number of hidden units.
            n_classes (int): Number of output units.
            lr (float): Learning rate.
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        # Overrides its parent class with any custom arguments if needed
        super(MLP, self).__init__(init_weights, device)

        # Number of inputs
        self.n_input = n_input

        # Linear layers
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

        # Compiles the network
        self._compile(lr)

    def forward(self, x):
        """Performs a forward pass over the architecture.

        Args:
            x (torch.Tensor): Tensor containing the data.

        Returns:
            The logits predictions over the input tensor.

        """

        # Arranges the input as a continuous amount of units
        x = x.view(-1, self.n_input)

        # Passes down the linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
