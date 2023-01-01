from typing import get_args, Literal

import numpy as np
import matplotlib.pyplot as plt

from src.si.statistics.sigmoid_function import sigmoid_function
from src.si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy

alpha_opt = Literal['static_alpha', 'half_alpha']


class LogisticRegression:
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 2000,
                 alpha_type: alpha_opt = 'static_alpha'):
        """
        Logistic Regression

        Parameters
        ----------
        :param l2_penalty: float
            Regularization coeficient L2
        :param alpha: float
            Learning rate
        :param max_iter: int
            Maximum number of iterations
        :param alpha_type: alpha_opt
            The gradient descent algorithm to use. There are two options: (1) 'static_alpha': where no alterations are
            applied to the alpha; or (2) 'half_alpha' where the value of alpha is set to half everytime the cost function
            value remains the same.

        Attributes
        ----------
        theta: np.ndarray
            the model coefficients/parameters for the input variables (features)
        theta_zero: float
            the coefficient/parameter zero. Also known as intercept
        cost_history: dict
            the cost history of the model
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.alpha_type = alpha_type

        # attributes
        self.theta = None  # model coefficient
        self.theta_zero = None  # f function of a linear model
        self.cost_history = None  # history of the cost function

    def fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset
        Estimates theta and theta_zero for input dataset

        Parameters
        ----------
        :param dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: LogisticRegression
            The fitted model
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # initialize the cost history dictionary
        self.cost_history = {}

        # asserts that the alpha_type is valid
        opts = get_args(alpha_opt)
        assert self.alpha_type in opts, f'alpha_type must be one of {opts}'

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = np.dot(dataset.x, self.theta) + self.theta_zero

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.x)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            # compute the cost
            self.cost_history[i] = self.cost(dataset)

            # condition to stop the gradient descent if the cost is not changing
            threshold = 0.0001

            if i > 1 and self.cost_history[i - 1] - self.cost_history[i] < threshold:
                if self.alpha_type == 'half_alpha':
                    self.alpha /= 2

                else:
                    break

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the labels of the dataset
        Estimates the output (dependent) variable using the estimated thetas
        Converts estimated values to 0 or 1 (binary). Values equal to or greater than 0.5 take the value of 1.
        Values less than 0.5 take the value of 0.

        Parameters
        ----------
        :param dataset: Dataset
            The dataset to predict the labels of

        Returns
        -------
        y_pred: np.array
            The accuracy of the model
        """
        preds = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)

        mask = preds >= 0.5
        preds[mask] = 1
        preds[~mask] = 0
        return preds

    def score(self, dataset: Dataset) -> float:
        """
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        :param dataset: Dataset
            The dataset to score the model on

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Calculates the cost function between forecasts and actual values

        Parameters
        ----------
        :param dataset: Dataset
            The dataset to calculate the cost of the model on

        Returns
        -------
        cost: float
            The cost of the model
        """
        predictions = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)
        cost = (-dataset.y * np.log(predictions)) - ((1 - dataset.y) * np.log(1 - predictions))
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * dataset.shape()[0]))
        return cost

    def cost_plot(self):
        """
        Plots the cost history of the model
        """
        plt.plot(list(self.cost_history.keys()), list(self.cost_history.values()))
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()
