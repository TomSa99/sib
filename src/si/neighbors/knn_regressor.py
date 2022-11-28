import numpy as np

from sib.src.si.statistics.euclidean_distance import euclidean_distance
from sib.src.si.data.dataset import Dataset
from sib.src.si.metrics import rmse
from typing import Union

class KNNClassifier:
    """
    KNNRegressor is indicated for regression problems, therefore,
    it estimates an average value of the k most similar examples.

    Parameters
    ----------
    :param k: int
        Number of neighbors to consider.
    :param distance: euclidean_distance
        Distance function to use.

    Attributes
    ----------
    dataset: np.ndarray
        The training dataset.
    """
    def __init__(self, k: int, distance: euclidean_distance):
        self.k = k
        self.distance = distance

        self.dataset = None

    def fit(self, dataset: Dataset):
        self.dataset = dataset
        return self

    def _get_closest_label (self, sample: np.ndarray) -> Union[int, str]:
        """
        Returns the closest label to the sample.

        Parameters
        ----------
        :param sample: np.ndarray
            The sample to classify.

        Returns
        -------
        :return: Union[int, str]
            The closest label.
        """
        distances = self.distance(sample, self.dataset.x)
        closest_labels = self.dataset.y[np.argsort(distances)[:self.k]]
        labels, counts = np.unique(closest_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the labels of the dataset.

        Parameters
        ----------
        :param dataset: Dataset
            The dataset to predict.

        Returns
        -------
        :return: np.ndarray
            The predicted labels.
        """
        return np.apply_along_axis(self._get_closest_label, 1, dataset.x)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model.

        Parameters
        ----------
        :param dataset: Dataset
            The dataset to test the model.

        Returns
        -------
        :return: float
            The accuracy of the model.
        """
        predicitons = self.predict(dataset)
        return rmse(dataset.y, predicitons)