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
        self.loss = nn.NLLLoss()

        # Check if there is a tuple for the weights initialization
        if init_weights:
            # Iterate over all possible parameters
            for _, p in self.named_parameters():
                # Initializes with a uniform distributed value
                nn.init.uniform_(p.data, init_weights[0], init_weights[1])

    def step(self, batch, train=True):
        """Performs a single batch optimization step.

        Args:
            batch (tuple): Tuple containing the batches input (x) and target (y).
            train (bool): Whether it is a training step or not.

        Returns:
            The loss and accuracy accross the batch.

        """

        # Gathers the batch's input and target
        x, y = batch[0], batch[1]

        # Calculate the predictions based on inputs
        preds = self(x)

        # Calculates the batch's loss
        batch_loss = self.loss(preds, y)

        # Checks if it is a training batch
        if train:
            # Propagates the gradients backward
            batch_loss.backward()

            # Perform the parameeters updates
            self.optimizer.step()

        # Calculates the batch's accuracy
        batch_acc = torch.mean((torch.sum(torch.argmax(preds, dim=1) == y).float()) / x.size(0))

        return batch_loss.item(), batch_acc.item()

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

            # Initializes both accuracies as zero
            train_acc, val_acc = 0.0, 0.0

            # For every batch in the iterator
            for batch in tqdm(train_iterator):
                # Resetting the gradients
                self.optimizer.zero_grad()

                # Calculates the training loss and training accuracy
                loss, acc = self.step(batch)

                # Summing up batch's loss
                train_loss += loss

                # Summing up batch's accuracy
                train_acc += acc

            # Gets the mean training loss across all batches
            train_loss /= len(train_iterator)

            # Gets the mean training accuracy across all batches
            train_acc /= len(train_iterator)

            print(f'Loss: {train_loss} | Accuracy: {train_acc}')

            # If there is a validation iterator
            if val_iterator:
                # Setting the evalution flag
                self.eval()

                # Inhibits the gradient from updating the parameters
                with torch.no_grad():
                    # For every batch in the iterator
                    for batch in tqdm(val_iterator):
                        # Calculates the validation loss
                        loss, acc = self.step(batch, train=False)

                        # Summing up batch's loss
                        val_loss += loss

                        # Summing up batch's accuracy
                        val_acc += acc

                # Gets the mean validation loss across all batches
                val_loss /= len(val_iterator)

                # Gets the mean training accuracy across all batches
                val_acc /= len(val_iterator)

                print(f'Val Loss: {val_loss} | Val Accuracy: {val_acc}')

    def evaluate(self, test_iterator):
        """Evaluates the model.

        Args:
            test_iterator (torchtext.data.Iterator): Testing data iterator.

        """

        print('Evaluating model ...')

        # Setting the evalution flag
        self.eval()

        # Initializes the loss and accuracy as zero
        test_loss, test_acc = 0.0, 0.0

        # Inhibits the gradient from updating the parameters
        with torch.no_grad():
            # For every batch in the iterator
            for i, batch in enumerate(test_iterator):
                # Calculates the test loss
                loss, acc = self.step(batch, train=False)

                # Summing up batch's loss
                test_loss += loss

                # Summing up batch's accuracy
                test_acc += acc

        # Gets the mean validation loss across all batches
        test_loss /= len(test_iterator)

        # Gets the mean testing accuracy across all batches
        test_acc /= len(test_iterator)

        print(f'Loss: {test_loss} | Accuracy: {test_acc}')
