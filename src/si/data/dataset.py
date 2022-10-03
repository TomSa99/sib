from pyexpat import features
from re import X
import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, x, y, features, label):
        self.x = x
        # a matriz/tabela de features (variáveis independentes)
        self.y = y
        # o vetor da variável dependente
        self.features = features
        # o vetor do nome das features
        self.label = label
        # o nome do vetor da variável dependente
    
    def shape(self):
        # dimensões do dataset
        return self.x.shape
    
    def has_label(self):
        # verfica se o dataset tem y
        return self.y is not None
        # devolve True ou False
    
    def get_classes(self):
        # devolve as classes do dataset (valores possíveis de y)
        return np.unique(self.y)
    
    def get_mean(self):
        # devolve média para cada variável dependente
        return np.mean(self.x, axis=0)

    def get_variance(self):
        # devolve variância para cada variável dependente
        return np.var(self.x, axis=0)

    def get_median(self):
        # devolve mediana para cada variável dependente
        return np.median(self.x, axis=0)

    def get_min(self):
        # devolve valor mínimo para cada variável dependente
        return np.min(self.x, axis=0)

    def get_max(self):
        # devolve valor máximo para cada variável dependente
        return np.max(self.x, axis=0)

    def summary(self):
        # devolve um pandas DataFrame com todas as métricas descritivas
        df=pd.DataFrame(columns=self.features)
        df.loc['mean']=self.get_mean()
        df.loc['variance']=self.get_variance()
        df.loc['median']=self.get_median()
        df.loc['min']=self.get_min()
        df.loc['max']=self.get_max()
        return df

# teste
if __name__ == '__main__':
    x = np.array([[1,2,3] , [1,2,3]])
    y = np.array ([1,2])
    features = ['A','B','C']
    label = 'y'
    dataset = Dataset (x, y, features, label )
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.summary())
    