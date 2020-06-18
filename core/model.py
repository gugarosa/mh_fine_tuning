import math

import torch
from torch import nn, optim
from tqdm import tqdm


class Model(torch.nn.Module):
    """A Model class is responsible for customly implementing neural network architectures.

    One can configure, if necessary, different properties or methods that
    can be used throughout all childs.

    """

    def __init__(self, init_weights=None, device='cpu'):
        """Initialization method.

        Args:
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        # Override its parent class
        super(Model, self).__init__()

        # Creates a cpu-based device property
        self.device = device

        # Checks if GPU is avaliable
        if torch.cuda.is_available() and device == 'cuda':
            # Uses CUDA in the whole class
            self.cuda()

        # Setting default tensor type to float
        torch.set_default_tensor_type(torch.FloatTensor)

    def _compile(self, init_weights):
        """Compiles the network by setting its optimizer, loss function and additional properties.

        Args:
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.

        """

        # Defining an optimizer
        self.optimizer = optim.Adam(self.parameters())

        # Defines the loss as usual
        self.loss = nn.CrossEntropyLoss()

        # Check if there is a tuple for the weights initialization
        if init_weights:
            # Iterate over all possible parameters
            for _, p in self.named_parameters():
                # Initializes with a uniform distributed value
                nn.init.uniform_(p.data, init_weights[0], init_weights[1])

    def step(self, batch):
        """Performs a single batch optimization step.

        Args:
            batch (tuple): Tuple containing the batches input (x) and target (y).

        Returns:
            The training loss accross the batch.

        """

        # Gathers the batch's input and target
        x, y = batch[0], batch[1]

        # Resetting the gradients
        self.optimizer.zero_grad()

        # Calculate the predictions based on inputs
        preds = self(x)

        # Reshaping the tensor's size without the batch dimension
        preds = preds[1:].view(-1, preds.shape[-1])

        # Reshaping the tensor's size without the batch dimension
        y = y[1:].view(-1)

        # Calculates the batch's loss
        batch_loss = self.loss(preds, y)

        # Propagates the gradients backward
        batch_loss.backward()

        # Perform the parameeters updates
        self.optimizer.step()

        return batch_loss.item()

    def val_step(self, batch):
        """Performs a single batch evaluation step.

        Args:
            batch (tuple): Tuple containing the batches input (x) and target (y).

        Returns:
            The validation loss accross the batch.

        """

        # Gathers the batch's input and target
        x, y = batch[0], batch[1]

        # Calculate the predictions based on inputs
        preds = self(x)

        # Reshaping the tensor's size without the batch dimension
        preds = preds[1:].view(-1, preds.shape[-1])

        # Reshaping the tensor's size without the batch dimension
        y = y[1:].view(-1)

        # Calculates the batch's loss
        batch_loss = self.loss(preds, y)

        return batch_loss.item()

    def fit(self, train_iterator, val_iterator=None, epochs=10):
        """Trains the model.

        Args:
            train_iterator (torchtext.data.Iterator): Training data iterator.
            val_iterator (torchtext.data.Iterator): Validation data iterator.
            epochs (int): The maximum number of training epochs.
            
        """

        print('Fitting model ...')

        # Iterate through all epochs
        for e in range(epochs):
            print(f'Epoch {e+1}/{epochs}')

            # Setting the training flag
            self.train()

            # Initializes both losses as zero
            train_loss, val_loss = 0.0, 0.0

            # Defines a `tqdm` variable
            with tqdm(total=len(train_iterator)) as t:
                # For every batch in the iterator
                for i, batch in enumerate(train_iterator):
                    # Calculates the training loss
                    train_loss += self.step(batch)
                     
                    # Updates the `tqdm` status
                    t.set_postfix(loss=train_loss / (i + 1))
                    t.update()

            # Gets the mean training loss accross all batches
            train_loss /= len(train_iterator)

            print(f'Loss: {train_loss} | PPL: {math.exp(train_loss)}')

            # If there is a validation iterator
            if val_iterator:
                # Setting the evalution flag
                self.eval()

                # Inhibits the gradient from updating the parameters
                with torch.no_grad():
                    # Defines a `tqdm` variable
                    with tqdm(total=len(train_iterator)) as t:
                        # For every batch in the iterator
                        for i, batch in enumerate(val_iterator):
                            # Calculates the validation loss
                            val_loss += self.val_step(batch)

                            # Updates the `tqdm` status
                            t.set_postfix(val_loss=val_loss / (i + 1))
                            t.update()

                # Gets the mean validation loss accross all batches
                val_loss /= len(val_iterator)

                print(f'Val Loss: {val_loss} | Val PPL: {math.exp(val_loss)}')

    def evaluate(self, test_iterator):
        """Evaluates the model.

        Args:
            test_iterator (torchtext.data.Iterator): Testing data iterator.

        """

        print('Evaluating model ...')

        # Setting the evalution flag
        self.eval()

        # Initializes the loss as zero
        test_loss = 0.0

        # Inhibits the gradient from updating the parameters
        with torch.no_grad():
            # For every batch in the iterator
            for i, batch in enumerate(test_iterator):
                # Calculates the test loss
                test_loss += self.val_step(batch)

        # Gets the mean validation loss accross all batches
        test_loss /= len(test_iterator)

        print(f'Loss: {test_loss} | PPL: {math.exp(test_loss)}')
