import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils.loader as l
import utils.objects as o
import utils.optimizer as opt
import utils.targets as t


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Optimizes a machine learning model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['cifar10', 'cifar100'])

    parser.add_argument('model_input', help='Path to saved model that will be inputted', type=str)

    parser.add_argument('layer_name', help='Layer identifier to be optimized')

    parser.add_argument('mh', help='Meta-heuristic identifier', choices=['pso'])

    parser.add_argument('-batch_size', help='Batch size', type=int, default=100)

    parser.add_argument('-bounds', help='Searching bounds', type=float, default=0.01)

    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=10)

    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=15)

    parser.add_argument('-shuffle', help='Whether data should be shuffled or not', type=bool, default=True)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    shuffle = args.shuffle
    seed = args.seed

    # Gathering model's variables
    model_input = args.model_input
    layer_name = args.layer_name
    batch_size = args.batch_size

    # Gathering optimization variables
    bounds = args.bounds
    n_agents = args.n_agents
    n_iterations = args.n_iter
    mh_name = args.mh
    mh = o.get_mh(mh_name).obj
    hyperparams = o.get_mh(args.mh).hyperparams

    # Loads the data
    train, val, test = l.load_dataset(name=dataset)

    # Creates the iterators
    train_iterator = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    val_iterator = DataLoader(val, batch_size=batch_size, shuffle=shuffle)
    test_iterator = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Loads pre-trained model
    model = torch.load(model_input)

    # Gathering weights from desired layer
    W = getattr(model, layer_name).weight.detach().cpu().numpy()

    # Defining lower and upper bounds, and number of variables
    lb = list(np.reshape(W - bounds, W.shape[0] * W.shape[1]))
    ub = list(np.reshape(W + bounds, W.shape[0] * W.shape[1]))
    n_variables = W.shape[0] * W.shape[1]

    # Defining the optimization task
    opt_fn = t.fine_tune(model, layer_name, val_iterator)

    # Running the optimization task
    history = opt.optimize(mh, opt_fn, n_agents, n_variables, n_iterations, lb, ub, hyperparams)

    # Saving history object
    history.save(f'{model_input}.history')

    # Reshaping `w` to appropriate size
    W_best = np.reshape(history.best_agent[-1][0], (W.shape[0], W.shape[1]))

    # Converting numpy to tensor
    W_best = torch.from_numpy(W_best).float()

    # Replacing the layer weights
    setattr(getattr(model, layer_name), 'weight', torch.nn.Parameter(W_best))

    # Evaluating the model
    model.evaluate(test_iterator)

    # Saving optimized model
    torch.save(model, f'{model_input}.optimized')
