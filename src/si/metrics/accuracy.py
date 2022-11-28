import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy of the model.
    Calculate the error according to the accuracy formula: (VN+VP) / (VN+VP+FP+FN)

    Parameters
    ----------
    y_true: np.ndarray
        true labels
    y_pred: np.ndarray
        predicted labels

    Returns
    -------
    accuracy: float
        accuracy score
    """
    return np.sum(y_true == y_pred) / len(y_true)
