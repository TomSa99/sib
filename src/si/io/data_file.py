import pandas as pd
import numpy as np
from numpy import genfromtxt
from si.data.dataset import Dataset
from numpy import savetxt


def read_data_file(filename, sep=',', label=False):
    """
    Reads a data file into a Dataset object

    :param filename: str
        Path to the file
    :param sep: str, optional
        Separator used in the file, by default None
    :param label: bool, optional
        If the file has a label column, by default False

    :return: Dataset
    """
    data = genfromtxt(filename, delimiter=sep)

    if label:
        x = data[:, :-1]
        y = data[:, -1]

    else:
        x = data
        y = None

    return Dataset(x, y)


def write_data_file(filename, dataset, sep=',', label=False):
    """
    Writes a Dataset object into a data file

    :param filename: str
        Path to the file
    :param dataset: Dataset
        Dataset object
    :param sep: str, optional
        Separator used in the file, by default None
    :param label: bool, optional
        If the file has a label column, by default False

    :return: None
    """
    if label:
        x = dataset.x
        y = dataset.y
        data = np.concatenate((x, y), axis=1)
        savetxt(filename, data, delimiter=sep)

    else:
        savetxt(filename, dataset.x, delimiter=sep)


# test
if __name__ == '__main__':
    dataset = read_data_file('data.csv', sep=',', label=True)
    write_data_file(dataset, 'data.txt', sep=',', label=True)
