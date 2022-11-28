import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It calculates Root Mean Squared Error (RMSE) metric.

    Parameters
    ----------
    :param y_true: np.ndarray
        True values.
    :param y_pred: np.ndarray
        Predicted values.

    Returns
    -------
    :return: float
        RMSE metric between true and predicted values.
    """
    return np.sqrt(np.sum(np.square(y_true - y_pred))/len(y_true))