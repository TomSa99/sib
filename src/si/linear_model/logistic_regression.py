import numpy as np
from src.si.statistics.sigmoid_function import sigmoid_function
from src.si.data.dataset import Dataset


class LogisticRegression:
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
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

        Attributes
        ----------
        theta:
            the model coefficients/parameters for the input variables (features)
        theta_zero:
            the coefficient/parameter zero. Also known as intercept
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None

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

        # gradient descent
        for _ in range(self.max_iter):
            # calculate the predicted values
            y_pred = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)

            # calculate the gradient
            gradient = np.dot(dataset.x.T, (y_pred - dataset.y)) / m
            gradient_zero = np.sum(y_pred - dataset.y) / m

            # update the model parameters
            self.theta -= self.alpha * (gradient + self.l2_penalty * self.theta)
            self.theta_zero -= self.alpha * gradient_zero

        return self

    def predict(self, dataset: Dataset) -> np.array:
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
            The predicted labels of the dataset
        """
        vals = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)

        if vals >= 0.5: # 0.5 because is the half value of the sigmoid function (between 0 and 1)
            return 1
        else:
            return 0

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
        return np.sum(y_pred == dataset.y) / len(dataset.y)

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
        m, _ = dataset.shape()
        y_pred = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)
        cost = -1 / m * np.sum(dataset.y * np.log(y_pred) + (1 - dataset.y) * np.log(1 - y_pred))
        return cost