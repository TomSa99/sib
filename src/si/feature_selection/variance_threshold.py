import numpy as np
from sib.src.si.data.dataset import Dataset


class VarianceThreshold:
    def __init__(self, threshold=0.0):
        # parâmetros
        self.threshold = threshold
        # parâmetros estimados/calculados
        self.variances = None
    
    def fit(self, dataset):
    # estima/calcula a variância de cada feature; retorna o self (ele próprio)
        self.variances = np.var(dataset.X, axis=0) # np.var calcula a variância de cada feature
        return self
    
    def transform(self, dataset: Dataset):
    # retorna um novo x com as features que possuem variância maior que o threshold
        X = dataset.X
        feature_mask = self.variances > self.threshold
        X = X[:, feature_mask]
        features = np.array(dataset.features)[feature_mask]
        return Dataset(X=X, y=dataset.y, features=list(features),label=dataset.label)
    
    def fit_transform(self, dataset: Dataset):
    # chama o fit e o transform
        return self.fit(dataset).transform(dataset)

# test
if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = VarianceThreshold()
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)