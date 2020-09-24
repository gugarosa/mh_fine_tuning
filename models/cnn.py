import torchvision as tv
from torch import nn

from core.model import Model


class ResNet(Model):
    """A ResNet class implements a ResNet18 learning architecture.

    """

    def __init__(self, n_input=None, n_hidden=None, n_classes=10, lr=0.001, init_weights=None, device='cpu'):
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
        super(ResNet, self).__init__(init_weights, device)

        # Loads base model from torchvision
        self.model = tv.models.resnet18()

        # Replaces first convolutional layer with smaller kernel, stride and padding
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Replaces fully-connected layer with proper number of classes
        self.model.fc = nn.Linear(512, n_classes)

        # Compiles the network
        self._compile(lr)

    def forward(self, x):
        """Performs a forward pass over the architecture.

        Args:
            x (torch.Tensor): Tensor containing the data.

        Returns:
            The logits predictions over the input tensor.

        """

        # Passes down the model
        x = self.model(x)
        
        return x
