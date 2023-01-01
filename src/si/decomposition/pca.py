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
        self.mean = np.mean(dataset.x, axis=0)
        centered_data = dataset.x - dataset.x.mean(axis=0)

        # calculate SVD
        u, s, vh = np.linalg.svd(centered_data, full_matrices=False)

        # infer the main components
        self.components = vh[:self.n_components]

        # infer the explained variance
        self.explained_variance = s ** 2 / (dataset.x.shape[0] - 1)
        return self

    def transform(self, dataset: Dataset):
        """
        calculates the reduced dataset using principal components

        :param dataset: Dataset
            Dataset object
        """
        V = self.components.T
        centered_data = dataset.x - dataset.x.mean(axis=0)
        return np.dot(centered_data, V)

    def fit_transform(self, dataset: Dataset):
        """
        Fits the data to the model and calculates the reduced dataset

        :param dataset: Dataset
            Dataset object
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == "__main__":
    X = np.array([[0, 2, 0, 3],
                  [0, 1, 4, 3],
                  [0, 1, 1, 3]])
    dataset = Dataset(X,
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    a = PCA(n_components=2)
    print(a.fit_transform(dataset=dataset))
