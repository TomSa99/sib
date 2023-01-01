from typing import Callable, Union

import numpy as np

from src.si.data.dataset import Dataset
from src.si.metrics import accuracy
from src.si.statistics.euclidean_distance import euclidean_distance

class KNNClassifier:
    """
    KNN Classifier
    The k-nearest neighbors algorithm estimates the class for a sample based on the k most similar examples.

    Parameters
    ----------
    :param k: int
        Number of neighbors to use
    :param distance: Callable
        Function that calculates the distance between sample and training dataset samples

    Attributes
    ----------
    dataset: np.ndarray
        Training dataset
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        Fit the model using dataset as training data.
        Stores the training dataset

        Parameters
        ----------
        :param dataset: Dataset
            Training dataset

        Returns
        -------
        :return: self
            KNNClassifier
        """
        self.dataset = dataset
        return self

    def _get_closest_neighbors(self, sample: np.ndarray) -> Union[int, str]:
        """
        Returns the closest labels to the sample

        Parameters
        ----------
        :param sample: np.ndarray
            Sample to find the closest neighbors to

        Returns
        -------
        :return label: Union[int, str]
            The closest lable to the sample
        """
        # compute the distance between the sample and all the dataset
        distances = self.distance(sample, self.dataset.x)

        # get the k closest neighbors
        closest_neighbors = np.argsort(distances)[:self.k]

        # get the labels of the k closest neighbors
        closest_neighbors_labels = self.dataset.y[closest_neighbors]

        # return the most common label
        labels, counts = np.unique(closest_neighbors_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the class labels for the provided dataset.

        Parameters
        ----------
        :param dataset: Dataset
            Dataset to predict th class of

        Returns
        -------
        :return: np.ndarray
            Prediction of the models
        """
        return np.apply_along_axis(self._get_closest_neighbors, axis=1, arr=dataset.x)

    def score(self, dataset: Dataset) -> float:
        """
        Calculates accuracy between actual values and predictions

        Parameters
        ----------
        :param dataset: Dataset
            Test dataset

        Returns
        -------
        :return: float
            calculating the error between predictions and actual values
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    X = np.array([[0, 2, 0, 3],
                  [0, 1, 4, 3],
                  [0, 1, 1, 3]])
    dataset_ = Dataset(X,
                       y=np.array([0, 1, 0]),
                       features=["f1", "f2", "f3", "f4"],
                       label="y")
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNClassifier(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')