import numpy as np
from si.data.dataset import Dataset

class SlectKBest:
    def __init__(self, score_func, k):
        self.score_func = score_func
        # função de análise da variância (f_classificiation)
        self.k = k
        # numero de features a selecionar
        self.F = None
        # o valor de F para cada feature estima pela função score_func
        self.p = None
        # o valor de p para cada feature estima pela função score_func
    
    def fit(self, dataset: Dataset):
    # estima o F e p para cada feature usando a scoring_func; retorna o self (ele próprio)
        self.F, self.p = self.score_func(dataset)
        # devolve o valor de F e p para cada feature estima pela função score_func
        return self

    def transform(self, dataset: Dataset):
    # seleciona as k features com valor de F mais alto e retorna o X selecionado
        idx = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idx]
        return Dataset(dataset.x[:, idx], dataset.y, features, dataset.label)
        # devolve um novo dataset com as k features selecionadas

    def fit_transform(self, dataset: Dataset):
    # corre o fit e depois o transform
        self.fit(dataset)
        return self.transform(dataset)