from sib.src.si.data.dataset import Dataset
from scipy.stats import f_oneway

def f_classificiation (dataset: Dataset):
    """
    Scoring function for classification problems. It computes one-way ANOVA F-value for the
    provided dataset. The F-value scores allows analyzing if the mean between two or more groups (factors)
    are significantly different. Samples are grouped by the labels of the dataset.

    Parameters
    ----------
    :param dataset: Dataset
        Labeled dataset

    Returns
    -------
    F: np.array, shape (n_features,)
        F-value for each feature
    p: np.array, shape (n_features,)
        p-value for each feature
    """
    classes = dataset.get_classes()
    # devolve as classes do dataset (valores possíveis de y)
    groups = [dataset.x[dataset.y == samples] for samples in classes]
    f, p = f_oneway(*groups)
    return f, p
