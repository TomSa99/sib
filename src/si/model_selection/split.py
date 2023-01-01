from typing import Tuple

import numpy as np

from src.si.data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """Split a dataset into train and test sets.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    test_size : float
        The proportion of the dataset to include in the test split.
    random_state : int
        The random seed used to generate the split.

    Returns
    -------
    train : Dataset
        The training dataset.
    test : Dataset
        The test dataset.
    """
    # set random state
    np.random.seed(random_state)

    # dataset size
    n_samples = dataset.shape()[0]

    # number of samples in the test set
    n_test = int(n_samples * test_size)

    # dataset permutations
    permut = np.random.permutation(n_samples)

    # samples in the test set
    test_idxs = permut[:n_test]

    # samples in the training set
    train_idxs = permut[n_test:]

    # get the training and testing datasets
    train = Dataset(dataset.x[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.x[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test