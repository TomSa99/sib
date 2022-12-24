import numpy as np

from src.si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy

class StackingClassifier:
    """
    Ensemble classifier to generate predictions through a majority vote of models.
    Those predictions are then used to train a final model inputted.

    Parameters
    ----------
    :param models: List of models to use
    :param final_model: Final model to use
    """
    def __init__(self, models: list, final_model):
        """
        Initialize the StackingClassifier

        :param models: List of models to use
        :param final_model: Final model to use
        """
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit all models to the dataset

        :param dataset: Dataset to fit to
        :return: self: StackingClassifier
        """

        # train each model
        for model in self.models:
            model.fit(dataset)

        # gets the models predictions
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # trains the final model
        self.final_model.fit(Dataset(dataset.x, np.array(predictions).T))

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Combines the previsions of each model with a voting system

        :param dataset: Dataset to predict
        :return: Predicted class labels
        """
        # gets the models predictions
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # predicts the final model
        return self.final_model.predict(Dataset(dataset.x, np.array(predictions).T))

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model

        :param dataset: Dataset to score
        :return: Accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))