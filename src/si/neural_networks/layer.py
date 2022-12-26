import numpy as np

from src.si.statistics.sigmoid_function import sigmoid_function

class Dense:
    """
    Parameters
    ----------
    input_size : int
        The number of input neurons.
    output_size : int
        The number of output neurons.

    Attributes
    ----------
    weights : numpy.ndarray
        The weights of the layer.
    bias : numpy.ndarray
        The bias of the layer.
    """
    def __init__(self, input_size, output_size):
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(1, output_size)
        self.y = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the layer.
        Returns a 2d array with shape (1, output_size).

        :param X: numpy.ndarray
            The input data.

        :return: numpy.ndarray
            The output of the layer.
        """
        return np.dot(X, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Performs a backward pass through the layer.


        :param error: numpy.ndarray
            The error of the layer.
        :param learning_rate: float
            The learning rate.

        :return: numpy.ndarray
            The error of the previous layer.
        """

        error_prev = np.dot(error, self.weights.T)

        # update weights and bias
        self.weights -= learning_rate * np.dot(self.y.T, error)
        self.bias -= learning_rate * np.sum(error, axis=0)

        return error_prev

class SigmoidActivation:
    """
    A sigmoid activation layer.
    """
    def __init__(self):
        # attributes
        self.y = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the probability of each class

        :param X: np.ndarray
            The input data.

        :return: np.ndarray
            The output of the layer.
        """
        return sigmoid_function(X)

    def backward(self, error: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Performs a backward pass through the layer.

        :param error: numpy.ndarray
            The error of the layer.
        :param learning_rate: float
            The learning rate.

        :return: numpy.ndarray
            The error of the previous layer.
        """
        return error * self.y * (1 - self.y)

class ReLUActivation:
    """
    A ReLU activation layer.
    """
    def __init__(self):
        # attributes
        self.y = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the probability of each class

        :param X: np.ndarray
            The input data.

        :return: np.ndarray
            The output of the layer.
        """
        return np.maximum(0, X)

    def backward(self, error: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Performs a backward pass through the layer.

        :param error: numpy.ndarray
            The error of the layer.
        :param learning_rate: float
            The learning rate.

        :return: numpy.ndarray
            The error of the previous layer.
        """
        return error * (self.y > 0)

class LinearActivation:
    """
    A linear activation layer.
    """
    def __init__(self):
        # attributes
        self.y = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the probability of each class

        :param X: np.ndarray
            The input data.

        :return: np.ndarray
            The output of the layer.
        """
        return X

    def backward(self, error: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Performs a backward pass through the layer.

        :param error: numpy.ndarray
            The error of the layer.
        :param learning_rate: float
            The learning rate.

        :return: numpy.ndarray
            The error of the previous layer.
        """
        return error

