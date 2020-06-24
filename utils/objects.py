from opytimizer.core.optimizer import Optimizer
from opytimizer.optimizers.swarm import pso

from models.mlp import MLP


class Model:
    """A Model class to help users in selecting distinct architectures from the command line.

    """

    def __init__(self, obj):
        """Initialization method.

        Args:
            obj (Optimizer): An Optimizer-child instance.
            hyperparams (dict): Meta-heuristic hyperparams.

        """

        # Creates a property to hold the class itself
        self.obj = obj


# Defines a model dictionary constant with the possible values
MODEL = dict(
    mlp=Model(MLP)
)


def get_model(name):
    """Gets a model by its identifier.

    Args:
        name (str): Model's identifier.

    Returns:
        An instance of the Model class.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return MODEL[name]

    # If object is not found
    except:
        # Raises a RuntimeError
        raise RuntimeError(f'Model {name} has not been specified yet.')


class MetaHeuristic:
    """A MetaHeuristic class to help users in selecting distinct meta-heuristics from the command line.

    """

    def __init__(self, obj, hyperparams):
        """Initialization method.

        Args:
            obj (Optimizer): An Optimizer-child instance.
            hyperparams (dict): Meta-heuristic hyperparams.

        """

        # Creates a property to hold the class itself
        self.obj = obj

        # Creates a property to hold the hyperparams
        self.hyperparams = hyperparams


# Defines a meta-heuristic dictionary constant with the possible values
META = dict(
    pso=MetaHeuristic(pso.PSO, dict(w=0.7, c1=1.7, c2=1.7))
)


def get_mh(name):
    """Gets a meta-heuristic by its identifier.

    Args:
        name (str): Meta-heuristic's identifier.

    Returns:
        An instance of the MetaHeuristic class.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return META[name]

    # If object is not found
    except:
        # Raises a RuntimeError
        raise RuntimeError(
            f'Meta-heuristic {name} has not been specified yet.')
