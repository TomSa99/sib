import numpy as np
from src.si.data.dataset import Dataset


class VarianceThreshold:
    """
    Variance Threshold feature selection
    Features with a training-set variance lower than this threshold will be removed from the dataset.

    Parameters
    ----------
    :param threshold: float, default=0.0
        The threshold value to use for feature selection.
        Features with a training-set variance lower than this threshold will be removed.

    Attributes
    ----------
    variances: array-like of shape (n_features,)
        The variance of each feature.
    """
    def __init__(self, threshold=0.0):
        # parameters
        self.threshold = threshold

        # attributes
        self.variances = None

    def fit(self, dataset: Dataset):
        """
        Fit the VarianceThreshold model according to the given training data and parameters.
        Estimates/calculates the variance of each feature; returns the self (itself)

        :param dataset: Dataset
            The dataset to fit.

        :return: self
        """
        self.variances = np.var(dataset.x, axis=0)  # np.var calculates the variance of each feature
        return self

    def transform(self, dataset: Dataset):
        """
        It removes all features whose variance doesn’t meet the threshold.
        Selects all features with variance greater than the threshold and returns the selected X.

        :param dataset: Dataset
            The dataset to transform.

        :return: Dataset
            The transformed dataset.
        """
        x = dataset.x
        feature_mask = self.variances > self.threshold
        x = x[:, feature_mask]
        features = np.array(dataset.features)[feature_mask]
        return Dataset(x=x, y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset):
        """
        Fits to data, then transform it.

        :param dataset: Dataset
            The dataset to fit and transform.
        :return: Dataset
            The transformed dataset.
        """
        return self.fit(dataset).transform(dataset)


# test
if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset(x=np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = VarianceThreshold()
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)
