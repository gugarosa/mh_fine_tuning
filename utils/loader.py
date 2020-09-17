import torch
import torchtext as tt
import torchvision as tv

# A constant used to hold a dictionary of possible image datasets
INAGE_DATASETS = {
    'cifar10': tv.datasets.CIFAR10,
    'cifar100': tv.datasets.CIFAR100,
}

# A constant used to hold a dictionary of possible text datasets
TEXT_DATASETS = {
    'imdb': tt.datasets.IMDB,
    'sst': tt.datasets.SST
}


def load_image_dataset(name='cifar10', val_split=0.2):
    """Loads an input image dataset.

    Args:
        name (str): Name of dataset to be loaded.
        val_split (float): Percentage of split for the validation set.

    Returns:
        Training, validation and testing sets of loaded dataset.
        
    """

    # Defining the torch seed
    torch.manual_seed(0)

    # Loads the training data
    train = IMAGE_DATASETS[name](root='./data', train=True, download=True,
                           transform=tv.transforms.ToTensor())

    # Splitting the training data into training/validation
    train, val = torch.utils.data.random_split(
        train, [int(len(train) * (1 - val_split)), int(len(train) * val_split)])

    # Loads the testing data
    test = IMAGE_DATASETS[name](root='./data', train=False, download=True,
                          transform=tv.transforms.ToTensor())

    return train, val, test


def load_text_dataset(name='imdb', val_split=0.2):
    """Loads an input text dataset.

    Args:
        name (str): Name of dataset to be loaded.
        val_split (float): Percentage of split for the validation set.

    Returns:
        Training, validation and testing sets of loaded dataset.
        
    """

    # Defining the torch seed
    torch.manual_seed(0)

    # Defining fields
    TEXT = tt.data.Field(lower=True, batch_first=True)
    LABEL = tt.data.LabelField(sequential=True, batch_first=True)

    # Checks if it IMDB dataset
    if name == 'imdb':
        # Loads the data
        _train, test = TEXT_DATASETS[name].splits(TEXT, LABEL, root='./data')

        # Splits a new validation set
        train, val = _train.split(split_ratio=(1-val_split))

    # If is SST dataset
    elif name == 'sst':
        # Loads the data
        train, val, test = TEXT_DATASETS[name].splits(TEXT, LABEL, root='./data')

    # Builds the vocabulary
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    return train, val, test, len(TEXT.vocab)
