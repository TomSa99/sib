from si.data.dataset import Dataset
import numpy as np


class SelectPercentile:
    def __init__(self, score_func, percentile):
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset):
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset):
        len_feats = len(dataset.features)
        percentile = int(len_feats * self.percentile)
        idx = np.argsort(self.F)[:percentile]
        features = np.array(dataset.features)[idx]
        return Dataset(dataset.x[:, idx], dataset.y, features)

    def fit_transform(self, dataset: Dataset):
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == "__main__":
    a = SelectPercentile(0.75)
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    a = a.fit_transform(dataset)
    print(dataset.features)
    print(a.features)
