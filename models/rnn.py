import torch
from torch import nn

from core.model import Model


class LSTM(Model):
    """An LSTM class implements a Long Short-Term Memory learning architecture.

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
        super(LSTM, self).__init__(init_weights, device)

        # Embedding layer
        self.emb = nn.Embedding(n_input, n_hidden)

        # Recurrent layer
        self.rnn = nn.LSTM(input_size=n_hidden, hidden_size=n_hidden)

        # Linear layer
        self.fc = nn.Linear(n_hidden, n_classes)

        # Compiles the network
        self._compile(lr)

    def forward(self, x):
        """Performs a forward pass over the architecture.

        Args:
            x (torch.Tensor): Tensor containing the data.

        Returns:
            The logits predictions over the input tensor.

        """

        # Passes down through the network
        x = self.emb(x)
        x = self.rnn(x)
        x = self.fc(x)
        
        return x
