import numpy as np

class VarianceThreshold:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.variances = None
    
    def fit(self, x):
    # estima/calcula a variância de cada feature; retorna o self (ele próprio)
        self.variances = np.var(x, axis=0) # np.var calcula a variância de cada feature
        return self
    
    def transform(self, x):
    # retorna um novo x com as features que possuem variância maior que o threshold
        return x[:, self.variances > self.threshold]
    
    def fit_transform(self, x):
    # chama o fit e o transform
        return self.fit(x).transform(x)