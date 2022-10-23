import numpy as np
# euclidean distance
def euclidean_distance(x, y):
    """
    Calculate the Euclidean distance between X and Y using the following formula:
    distance_y1n = np.sqrt((x1 - y11)^2 + (x2 - y12)^2 + ... + (xn - y1n)^2)
    distance_y2n = np. sqrt((x1 - y21)^2 + (x2 - y22)^2 + ... + (xn - y2n)^2)
    ...

    Parameters
    ----------
    :param x: np.ndarray
        Point
    :param y: np.ndarray
        Set of points
    """
    return np.sqrt((x - y)**2).sum(axis=1) 
    # axis=1 sums rows, axis=0 sums columns