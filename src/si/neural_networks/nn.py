from typing import Callable
import numpy as np

from src.si.metrics.mse import mse, mse_derivative
from src.si.metrics.accuracy import accuracy
from src.si.data.dataset import Dataset

class NN:
    """
    Parameters
    ----------
    layers : list
        The layers of the neural network.
    epochs : int
        The number of epochs.
    learning_rate : float
        The learning rate.
    loss : Callable
        The loss function.
    loss_derivative : Callable
        The derivative of the loss function.
    verbose : bool
        If True, it prints the loss after each epoch.

    Attributes
    ----------
    history : Dict
    """
    def __init__(self,
                 layers: list,
                 epochs: int = 1000,
                 learning_rate: float = 0.01,
                 loss: Callable = mse,
                 loss_derivative: Callable = mse_derivative,
                 verbose: bool = False):
        # parameters
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.verbose = verbose

        # attributes
        self.history = {}

    def fit(self, dataset: Dataset) -> 'NN':
        """
        Trains the neural network.

        :param dataset: Dataset
            The dataset to train the model on.

        :return: NN
            The trained neural network.
        """
        x = dataset.x
        y = dataset.y

        for epoch in range(1, self.epochs + 1):
            # forward propagation
            for layer in self.layers:
                x = layer.forward(x)

            # backward propagation
            error = self.loss_derivative(y, x)
            for layer in self.layers[::-1]:
                error = layer.backward(error, self.learning_rate)

            # save history of loss
            cost = self.loss(y, x)
            self.history[epoch] = cost

            # print loss
            if self.verbose:
                print(f'Epoch: {epoch}/{self.epochs} --- Cost: {cost}')

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the outputs of the dataset.

        :param dataset: Dataset
            The dataset to predict the labels of.

        :return: numpy.ndarray
            The predicted labels.
        """
        x = dataset.x

        for layer in self.layers:
            x = layer.forward(x)

        return x

    def cost(self, dataset: Dataset) -> float:
        """
        Calculates the cost of the model on the given dataset.

        :param dataset: Dataset
            The dataset to calculate the cost of.

        :return: float
            The cost of the neural network.
        """
        pred = self.predict(dataset)
        return self.loss(dataset.y, pred)

    def score(self, dataset: Dataset, scoring_func: Callable = accuracy):
        """
        Calculates the score of the model on the given dataset.

        :param dataset: Dataset
            The dataset to calculate the score of.
        :param scoring_func: Callable
            The scoring function.

        :return: float
            The score of the neural network.
        """
        pred = self.predict(dataset)
        return scoring_func(dataset.y, pred)