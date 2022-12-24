import numpy as np

from src.si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy

class VotingClassifier:
    """
    Ensemble classifier that uses the majority vote to predict the class labels

    Parameters
    ----------
    :param models: List of models to use
    """
    def __int__(self, models: list):
        """
        Initialize the VotingClassifier

        :param models: List of models to use
        """
        self.models = models

    def fit(self, dataset: Dataset) -> 'VotingClassifier':
        """
        Fit all models to the dataset

        :param dataset: Dataset to fit to
        :return: self: VotingClassifier
        """
        for model in self.models:
            model.fit(dataset)
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Combines the previsions of each model with a voting system

        :param dataset: Dataset to predict
        :return: Predicted class labels
        """
        predictions = np.array([model.predict(dataset) for model in self.models])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        # np.bincount Counts the number of occurrences of each value in array of non-negative ints, giving us
        # the number of votes for each class, then the argmax returns the class with the most votes

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model

        :param dataset: Dataset to score
        :return: Accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))