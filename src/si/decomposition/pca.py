import numpy as np
from src.si.data.dataset import Dataset


class PCA:
    """
    Linear algebra technique to reduce the dimensions of the dataset.
    The PCA implementation uses a linear algebra technique SVD (Singular Value Decomposition)

    Parameters
    ----------
    :param n_components: int
        Number of components

    Attributes
    ----------
    mean: np.ndarray
        Samples mean
    components: np.ndarray
        The main components aka unit matrix of eigenvectors
    explained_variance: np.ndarray
        The explained variance aka diagonal matrix of the eigenvalues
    """

    def __init__(self, n_components: int):
        # parameters
        self.n_components = n_components

        # attributes
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset):
        """
        Fits the data to the model
        Estimates the mean, components, and explained variance

        :param dataset: Dataset
            Dataset object
        """
        # center the data
        mean = np.mean(dataset.data, axis=0)
        centered_data = dataset.data - mean

        # calculate SVD
        u, s, vh = np.linalg.svd(centered_data, full_matrices=False)
        x = u*s*vh

        # infer the main components
        components = vh.T

        # infer the explained variance
        explained_variance = s**2 / (dataset.data.shape[0] - 1)

    def transform(self, dataset: Dataset):
        """
        calculates the reduced dataset using principal components

        :param dataset: Dataset
            Dataset object
        """
        # center the data
        centered_data = dataset.data - self.mean

        # calculate the reduced dataset
        reduced_data = np.dot(centered_data, self.components)

