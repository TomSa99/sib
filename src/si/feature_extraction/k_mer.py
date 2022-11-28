from src.si.data.dataset import Dataset
from intertools import product
import numpy as np


class Kmer:
    """
    k-mer consists of set of substrings of length k contained in a sequence.

    Parameters
    ----------
    :param k: int
        Length of the substring

    Attributes
    ----------
    k_mers: set
        Set of substrings of length k contained in a sequence
    """

    def __init__(self, k):
        self.k = k

        self.k_mers = list(product('ACGT', repeat=k))

    def fit(self, dataset: Dataset):
        """
        estimates all possible k-mers; returns the self (itself)

        """
        self.k_mers = [''.join(k_mer) for k_mer in self.k_mers]
        return self.k_mers

    def _get_sequence_k_mer_composition(self, sequence: str):
        # calculate the k-mer composition
        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            counts[k_mer] += 1

    def transform(self, dataset: Dataset):
        """
        Transforms the data

        :param dataset: Dataset
            Dataset object
        """
        # calculate the k-mer composition
        sequence_k_mer_composition = [self._get_sequence_k_mer_composition(sequence) for sequence in dataset.x[:, 0]]

        sequence_k_mer_composition = np.array(sequence_k_mer_composition)

        return Dataset(x=sequence_k_mer_composition, y=dataset.y, features=self.k_mers, label=dataset.label)
