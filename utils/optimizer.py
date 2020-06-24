from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.spaces.search import SearchSpace


def optimize(opt, target, n_agents, n_iterations, hyperparams):
    """Abstracts all Opytimizer's mechanisms into a single method.

    Args:
        opt (Optimizer): An Optimizer-child class.
        target (callable): The method to be optimized.
        n_agents (int): Number of agents.
        n_iterations (int): Number of iterations.
        hyperparams (dict): Dictionary of hyperparameters.

    Returns:
        A History object containing all optimization's information.
        
    """

    # Creating the SearchSpace
    space = SearchSpace(n_agents=n_agents, n_variables=1,
                        n_iterations=n_iterations, lower_bound=[0.00001], upper_bound=[1])

    # Creating the Optimizer
    optimizer = opt(hyperparams=hyperparams)

    # Creating the Function
    function = Function(pointer=target)

    # Creating the optimization task
    task = Opytimizer(space=space, optimizer=optimizer, function=function)

    return task.start(store_best_only=True)
