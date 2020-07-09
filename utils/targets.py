import numpy as np
import torch


def fine_tune(model, layer_name, val_iterator):
    """Wraps the reconstruction task for optimization purposes.

    Args:
        model (Model): Child object from Model class.
        layer_name (str): Identifier of the layer to be optimized.
        val_iterator (torchtext.data.Iterator): Validation data iterator.

    """

    def f(w):
        """Gathers weights from the meta-heuristic and evaluates over validation data.

        Args:
            w (float): Array of variables.

        Returns:
            1 - accuracy.

        """

        # Gathering model's weights
        W = getattr(model, layer_name).weight

        # Reshaping optimization variables to appropriate size
        W_cur = np.reshape(w, (W.size(0), W.size(1)))

        # Converting numpy to tensor
        W_cur = torch.from_numpy(W_cur).float()

        # Replacing the layer weights
        setattr(getattr(model, layer_name), 'weight', torch.nn.Parameter(W_cur))

        # Evaluating its validation accuracy
        _, acc = model.evaluate(val_iterator, validation=True)

        return 1 - acc

    return f
