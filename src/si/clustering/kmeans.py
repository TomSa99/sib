# O algoritmo k-means agrupa amostras em grupos chamados centroids. 
# O algoritmo tenta reduzir a distância entre as amostras e o centroid.

from typing import Callable
import numpy as np
from sympy import centroid

from sib.src.si.statistics.euclidean_distance import euclidean_distance


class KMeans:
    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = 'euclidean_distance'):
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        self.centroids = None
        self.labels = None
    
    def _init_centroids(self, dataset: Dataset):
        # inicializa os centroids
        seeds = np.random.permutation(dataset.x.shape[0])[:self.k] # uma amostra em cada centroid
        self.centroids = dataset.x[seeds] # seeds = amostra 5 / amostra 10 / amostra 15...
        pass

    def _get_closest_centroid(self, sample: np.ndarray):
        # calcula a distância entre as amostras e os centroids
        centroids_distance = self.distance(sample, self.centroids)
        # calcula o index do centroid mais próximo da amostra
        closest_centroids_index = np.argmin(centroids_distance, axis = 0)
        return closest_centroids_index


    def fit (self, k):
        # infere os centroids minimizando a distância entre as amostras e o centroid
        new_labels = np.apply_along_axis(self._get_closest_centroid, axis = 1,arr = dataset.x)
        
        centroids = []
        for i in range(self.k):
            centroid = np.mean(dataset.x[new_labels == i], axis=0)
            centroids.append(centroid)
        self.centroids = np.array(centroids)


        pass

    def predict(self, dataset: Dataset) -> np.ndarray:
        # infere qual dos centroids está mais perto da amostra
        return np.apply_along_axis(self._get_closest_centroid, axis = 1, arr = dataset.x)
        pass

    def transform(self,k):
        # calcula as distâncias entre as amostras e os centroids
        centroids_distance = np.apply_along_axis(self.)
        pass

    def predict(self, k):
        # infere qual dos centroids está mais perto da amostra
        pass