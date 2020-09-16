import torch
from torch import nn

from core.model import Model


class LSTM(Model):
    """An LSTM class implements a Long Short-Term Memory learning architecture.

    """

    def __init__(self, n_input=784, n_embedding=256, n_hidden=128, n_classes=5, lr=0.001, init_weights=None, device='cpu'):
        """Initialization method.

        Args:
            n_input (int): Number of input units.
            n_embedding (int): Number of embedding units.
            n_hidden (int): Number of hidden units.
            n_classes (int): Number of output units.
            lr (float): Learning rate.
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        # Overrides its parent class with any custom arguments if needed
        super(LSTM, self).__init__(init_weights, device)

        # Embedding layer
        self.emb = nn.Embedding(n_input, n_embedding)

        # Recurrent layer
        self.rnn = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden, batch_first=True)

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
        out, (h, c) = self.rnn(x)
        preds = self.fc(h[-1])
        
        return preds
