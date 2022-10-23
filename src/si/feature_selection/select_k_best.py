import numpy as np
from si.data.dataset import Dataset


class SlectKBest:
    """
    Select features according to the k highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    :param score_func : callable
        Function taking a dataset and returning a pair of arrays (scores, p_values)
    :param k : int, default=10
        Number of top features to select.

    Attributes
    ----------
    F : array, shape (n_features)
        The F-value of each feature.

    p : array, shape (n_features)
        The p-value of each F-score.
    """

    def __init__(self, score_func, k):
        # parameters
        # analysis of variance function (f_classificiation)
        self.score_func = score_func
        # number of features to select
        self.k = k

        # attributes
        # the value of F for each feature estimated by the score_func function
        self.F = None
        # the value of p for each feature estimated by the score_func function
        self.p = None

    def fit(self, dataset: Dataset):
        """
        It fits SelectKBest to compute the F scores and p values.

        :param dataset: Dataset
            A labeled dataset.

        :return: object
            Returns self.
        """
        # estimates the F and p for each feature using the scoring_func; returns the self (itself)
        self.F, self.p = self.score_func(dataset)
        # returns the value of F and p for each feature estimated by the score_func function
        return self

    def transform(self, dataset: Dataset):
        """
        It transforms the dataset by selecting the k best features.

        :param dataset: Dataset
            A labeled dataset

        :return: Dataset
            A labeled dataset with the k best features.
        """
        # selects the k features with the highest F value and returns the selected X
        idx = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idx]
        return Dataset(dataset.x[:, idx], dataset.y, features, dataset.label)

    def fit_transform(self, dataset: Dataset):
        """
        It fits and transforms the dataset by selecting the k best features.

        :param dataset: Dataset
            A labeled dataset

        :return: Dataset
            A labeled dataset with the k best features.
        """
        # corre o fit e depois o transform
        self.fit(dataset)
        return self.transform(dataset)
