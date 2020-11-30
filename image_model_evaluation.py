import argparse

import torch
from torch.utils.data import DataLoader

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Evaluates an image-based machine learning model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['cifar10', 'cifar100'])

    parser.add_argument('model_input', help='Path to saved model that will be evaluated', type=str)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=100)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    parser.add_argument('--shuffle', help='Whether data should be shuffled or not', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    seed = args.seed
    shuffle = args.shuffle

    # Gathering model's variables
    model_input = args.model_input
    batch_size = args.batch_size

    # Loads the data
    _, _, test = l.load_image_dataset(name=dataset, seed=seed)

    # Creates the iterators
    test_iterator = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Loads pre-trained model
    model = torch.load(model_input)

    # Evaluating the model
    model.evaluate(test_iterator)
