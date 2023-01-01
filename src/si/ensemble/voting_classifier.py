import numpy as np

from typing import List

from src.si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy


class VotingClassifier:
    """
    Ensemble classifier that uses the majority vote to predict the class labels

    Parameters
    ----------
    :param models: List of models to use
    """

    def __init__(self, models: List):
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

    def predict(self, dataset: Dataset) -> np.array:
        """
        Combines the previsions of each model with a voting system

        :param dataset: Dataset to predict
        :return: Predicted class labels
        """
        def _get_most_voted(pred: np.array) -> int:
            labels, counts = np.unique(pred, return_counts=True)

            return labels[np.argmax(counts)]

        # list of the predictions
        predictions = []

        for model in self.models:
            predictions.append(model.predict(dataset))

        predictions = np.array(predictions)
        most_voted = np.apply_along_axis(_get_most_voted, axis=0, arr=predictions)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model

        :param dataset: Dataset to score
        :return: Accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))


# if __name__ == '__main__':
#     # import dataset
#     from src.si.data.dataset import Dataset
#     from src.si.model_selection.split import train_test_split
#     from src.si.neighbors.knn_classifier import KNNClassifier
#     from src.si.linear_model.logistic_regression import LogisticRegression
#
#     # load and split the dataset
#     X = np.array([[0, 2, 0, 3],
#                   [0, 1, 4, 3],
#                   [0, 1, 1, 3]])
#     dataset_ = Dataset(X,
#                        y=np.array([0, 1, 0]),
#                        features=["f1", "f2", "f3", "f4"],
#                        label="y")
#     dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)
#
#     # initialize the KNN and Logistic classifier
#     knn = KNNClassifier(k=3)
#     lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
#
#     # initialize the Voting classifier
#     voting = VotingClassifier([knn, lg])
#
#     voting.fit(dataset_train)
#
#     # compute the score
#     score = voting.score(dataset_test)
#     print(f"Score: {score}")
#
#     print(voting.predict(dataset_test))
