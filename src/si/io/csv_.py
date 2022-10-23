import pandas as pd
import numpy as np
from si.data.dataset import Dataset


# pandas.read_csv(https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

# pd.read_csv('/Users/Utilizador/Desktop/SIB/notas.csv', sep=',')

def read_csv(filename, sep=',', features=False, label=False):
    """
    read csv file

    :param filename: str
        Path to the csv file
    :param sep: str, optional
        The separator used in the file. Defaults to ','.
    :param features: bool, optional
        If True, the file has a header. Defaults to False.
    :param label: bool, optional
        If True, the file has a label. Defaults to False.

    :return: Dataset
        Dataset object
    """
    df = pd.read_csv(filename, sep=sep)

    if features and label:
        features = df.columns[:-1]
        label = df.columns[-1]
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()

    elif features and not label:
        features = df.columns
        x = df.to_numpy()
        y = None

    elif not features and label:
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        features = None
        label = None

    else:
        x = df.to_numpy()
        y = None
        features = None
        label = None

    return Dataset(x, y, features, label)


def write_csv(filename, dataset, sep=',', features=False, label=False):
    """
    write a dataset object to a csv file

    :param filename: str
        Path to the csv file
    :param dataset: Dataset
        Dataset object
    :param sep: str, optional
        The separator used in the file. Defaults to ','.
    :param features: bool, optional
        If True, the file has a header. Defaults to False.
    :param label: bool, optional
        If True, the file has a label. Defaults to False.

    :return: None
    """
    if features and label:
        df = pd.DataFrame(dataset.x, columns=dataset.features)
        df[dataset.labels] = dataset.y
        df.to_csv(filename, sep=sep, index=False)

    elif features:
        df = pd.DataFrame(dataset.x, columns=dataset.features)
        df.to_csv(filename, sep=sep, index=False)

    elif label:
        df = pd.DataFrame(dataset.y, columns=dataset.labels)
        df.to_csv(filename, sep=sep, index=False)

    else:
        df = pd.DataFrame(dataset.x)
        df.to_csv(filename, sep=sep, index=False)


"testar o código"
if __name__ == '__main__':
    # read_csv
    dataset = read_csv('/Users/Utilizador/Desktop/SIB/notas.csv', sep=',', features=True, label=True)
    print(dataset.summary())

    # write_csv
    write_csv('/Users/Utilizador/Desktop/SIB/notas2.csv', dataset, sep=',', features=True, label=True)

read_csv('/Users/Utilizador/Desktop/SIB/notas.csv')
