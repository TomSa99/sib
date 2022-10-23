from pyexpat import features
from re import X
import numpy as np
import pandas as pd


class Dataset:
    """
    Represents a machine learning tabular dataset

    Parameters
    ----------
    :param x: numpy.ndarray (n_samples, n_features)
        The matrix of features (independent variables)
    :param y: numpy.ndarray (n_samples,1)
        The vector of the dependent variable (labels)
    :param features: list (n_features)
        The list of feature names
    :param label: str
        The name of the dependent variable (label)
    """
    def __init__(self, x, y, features, label):
        # the feature matrix/table (independent variables)
        self.x = x
        # the vector of the dependent variable (labels)
        self.y = y
        # the feature name vector
        self.features = features
        # the vector name of the dependent variable
        self.label = label

    def shape(self):
        """
        Returns the shape of the dataset

        :return: tuple (n_samples, n_features)
        """
        return self.x.shape

    def has_label(self):
        """
        Returns True if the dataset has labels

        :return: bool
        """
        return self.y is not None

    def get_classes(self):
        """
        Returns the unique classes in the dataset

        :return: numpy.ndarray (n_classes)
        """
        return np.unique(self.y)

    def get_mean(self):
        """
        Returns the mean of each feature

        :return: numpy.ndarray (n_features)
        """
        return np.mean(self.x, axis=0)

    def get_variance(self):
        """
        Returns the variance of each feature

        :return: numpy.ndarray (n_features)
        """
        return np.var(self.x, axis=0)

    def get_median(self):
        """
        Returns the median of each feature

        :return: numpy.ndarray (n_features)
        """
        return np.median(self.x, axis=0)

    def get_min(self):
        """
        Returns the minimum of each feature

        :return: numpy.ndarray (n_features)
        """
        return np.min(self.x, axis=0)

    def get_max(self):
        """
        Returns the maximum of each feature

        :return: numpy.ndarray (n_features)
        """
        return np.max(self.x, axis=0)

    def summary(self):
        """
        Returns a summary of the dataset

        :return: pandas.DataFrame (n_features, 5)
        """
        df = pd.DataFrame(columns=self.features)
        df.loc['mean'] = self.get_mean()
        df.loc['variance'] = self.get_variance()
        df.loc['median'] = self.get_median()
        df.loc['min'] = self.get_min()
        df.loc['max'] = self.get_max()
        return df

    def dropna(self):
        """
        Drops all rows with missing values

        :return: Dataset
        """
        self.x = self.x[~np.isnan(self.x).any(axis=1)]

        # update vector y removing entries associated with samples to be removed
        if self.has_label():
            self.y = self.y[~np.isnan(self.x).any(axis=1)]

    def fillna(self, value):
        """
        Fills all missing values with the given value

        :param value: float
            The value to fill the missing values

        :return: Dataset
        """
        self.x = np.nan_to_num(self.x, nan=value)

        # update y
        if self.has_label():
            self.y = np.nan_to_num(self.y, nan=value)


# teste
if __name__ == '__main__':
    x = np.array([[1, 2, 3], [1, 2, 3]])
    y = np.array([1, 2])
    features = ['A', 'B', 'C']
    label = 'y'
    dataset = Dataset(x, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.summary())
