import argparse

import torch
from torchtext.data import BucketIterator

import utils.loader as l
import utils.objects as o


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Trains and evaluates a text-based machine learning model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['sst'])

    parser.add_argument('model_name', help='Model identifier', choices=['lstm'])

    parser.add_argument('model_output', help='Identifier to saved model', type=str)

    parser.add_argument('-n_embedding', help='Number of embedding units', type=int, default=256)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-n_class', help='Number of classes', type=int, default=5)

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
    output = args.model_output
    n_embedding = args.n_embedding
    n_hidden = args.n_hidden
    n_class = args.n_class
    lr = args.lr
    device = args.device
    batch_size = args.batch_size
    shuffle = args.shuffle
    epochs = args.epochs
    seed = args.seed

    # Loads the data
    train, val, test, n_input = l.load_text_dataset(name=dataset)

    # Creates the iterators
    train_iterator = BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                    device=device, sort=True, sort_within_batch=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                    device=device, sort=True, sort_within_batch=True)
    test_iterator = BucketIterator(test, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                    device=device, sort=True, sort_within_batch=True)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Gathering the model
    model_obj = o.get_model(name).obj

    # Initializing the model
    model = model_obj(n_input=n_input, n_embedding=n_embedding, n_hidden=n_hidden,
                      n_classes=n_class, lr=lr, init_weights=None, device=device)

    # Fitting the model
    model.fit(train_iterator, val_iterator, epochs=epochs)

    # Evaluating the model
    model.evaluate(test_iterator)

    # Saving model
    torch.save(model, output)
