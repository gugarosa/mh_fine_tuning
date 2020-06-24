import numpy as np
import torch


def fine_tune(model, val_iterator):
    """Wraps the reconstruction task for optimization purposes.

    Args:

    """

    def f(w):
        """Gathers weights from the meta-heuristic and evaluates over validation data.

        Args:
            w (float): Array of variables.

        Returns:
            1 - accuracy.

        """

        # Reshaping `w` to appropriate size
        w = np.reshape(w, (model.fc2.weight.size(0), model.fc2.weight.size(1)))

        # Converting numpy to tensor
        w = torch.from_numpy(w).float()

        # Replacing the layer weights
        model.fc2.weight = torch.nn.Parameter(w)

        # Evaluating its validation accuracy
        _, acc = model.evaluate(val_iterator)

        return 1 - acc

    return f
