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
    parser = argparse.ArgumentParser(usage='Trains and evaluates an image-based machine learning model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['cifar10', 'cifar100'])

    parser.add_argument('model_name', help='Model identifier', choices=['mlp', 'resnet'])

    parser.add_argument('model_output', help='Identifier to saved model', type=str)

    parser.add_argument('-n_input', help='Number of input units', type=int, default=3072)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-n_class', help='Number of classes', type=int, default=10)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=100)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=10)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'])

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    parser.add_argument('--shuffle', help='Whether data should be shuffled or not', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    name = args.model_name
    output = args.model_output
    n_input = args.n_input
    n_hidden = args.n_hidden
    n_class = args.n_class
    lr = args.lr
    device = args.device
    batch_size = args.batch_size
    epochs = args.epochs
    seed = args.seed
    shuffle = args.shuffle

    # Loads the data
    train, _, _ = l.load_image_dataset(name=dataset, seed=seed)

    # Creates the iterators
    train_iterator = DataLoader(train, batch_size=batch_size, shuffle=shuffle)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Gathering the model
    model_obj = o.get_model(name).obj

    # Initializing the model
    model = model_obj(n_input=n_input, n_hidden=n_hidden, n_classes=n_class, lr=lr, init_weights=None, device=device)

    # Fitting the model
    model.fit(train_iterator, epochs=epochs)

    # Saving model
    torch.save(model, output)
