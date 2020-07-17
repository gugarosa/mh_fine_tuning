import argparse

import torch
from torch.utils.data import DataLoader

import utils.loader as l
import utils.objects as o


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Trains and evaluates a machine learning model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['cifar10', 'cifar100'])

    parser.add_argument('model_name', help='Model identifier', choices=['mlp'])

    parser.add_argument('-n_input', help='Number of input units', type=int, default=3072)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-n_class', help='Number of classes', type=int, default=10)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=100)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=10)

    parser.add_argument('-shuffle', help='Whether data should be shuffled or not', type=bool, default=True)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'])

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    name = args.model_name
    n_input = args.n_input
    n_hidden = args.n_hidden
    n_class = args.n_class
    lr = args.lr
    device = args.device
    batch_size = args.batch_size
    shuffle = args.shuffle
    epochs = args.epochs
    seed = args.seed

    # Loads the data
    train, val, test = l.load_dataset(name=dataset)

    # Creates the iterators
    train_iterator = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    val_iterator = DataLoader(val, batch_size=batch_size, shuffle=shuffle)
    test_iterator = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Gathering the model
    model_obj = o.get_model(name).obj

    # Initializing the model
    model = model_obj(n_input=n_input, n_hidden=n_hidden, n_classes=n_class, lr=lr, init_weights=None, device=device)

    # Fitting the model
    model.fit(train_iterator, val_iterator, epochs=epochs)

    # Evaluating the model
    model.evaluate(test_iterator)
