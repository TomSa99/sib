import numpy as np

def sigmoid_function(x):
    """
    Sigmoid function
    :param x: input

    :return: output
        The probability of the values being equal to 1
    """
    return 1 / (1 + np.exp(-x))