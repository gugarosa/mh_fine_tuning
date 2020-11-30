import argparse

import torch
from sklearn.metrics import classification_report
from torchtext.data import BucketIterator

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Evaluates a text-based machine learning model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['imdb', 'sst'])

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
    _, _, test, _ = l.load_text_dataset(name=dataset, seed=seed)

    # Creates the iterator
    test_iterator = BucketIterator(test, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                   sort=True, sort_within_batch=True)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Loads pre-trained model
    model = torch.load(model_input)

    # Predicts test data with the model
    y_preds, y_true = model.predict(test_iterator)

    # Performs the classification report
    report = classification_report(y_true, y_preds, output_dict=True)

    print(report)
