import numpy as np

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the cross-entropy loss.

    :param y_true: The true labels.
    :param y_pred: The predicted labels.

    Returns: The cross-entropy loss.
    """
    return -np.sum(y_true * np.log(y_pred))

def cross_entropy_derivativy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculates the cross-entropy loss derivative.

    :param y_true: The true labels.
    :param y_pred: The predicted labels.

    Returns: The cross-entropy loss derivative.
    """
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / len(y_true)