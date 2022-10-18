# O algoritmo k-means agrupa amostras em grupos chamados centroids. 
# O algoritmo tenta reduzir a distância entre as amostras e o centroid.

from typing import Callable
import numpy as np
from sympy import centroid

from src.si.statistics.euclidean_distance import euclidean_distance
from src.si.data.dataset import Dataset


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

    def _get_closest_centroid(self, sample: np.ndarray):
        # calcula a distância entre as amostras e os centroids
        centroids_distance = self.distance(sample, self.centroids)
        # calcula o index do centroid mais próximo da amostra
        closest_centroids_index = np.argmin(centroids_distance, axis = 0)
        return closest_centroids_index


    def fit (self, dataset: Dataset):
        # infere os centroids minimizando a distância entre as amostras e o centroid
        convergence = False
        j = 0
        labels = np.zeros(dataset.x.shape()[0])
        while not convergence and j < self.max_iter:

            # get closest centroids
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis = 1,arr = dataset.x)
            
            # compute new centroids
            centroids = []
            for i in range(self.k):
                centroid = np.mean(dataset.x[new_labels == i], axis=0)
                centroids.append(centroid)
            self.centroids = np.array(centroids)

            # verifica se os centroids convergiram
            convergence = np.any(labels != new_labels)

            # atualiza os labels
            labels = new_labels

            # incrementa o contador
            j += 1
        self.labels = labels
        return self

    def _get_distances(self, sample: np.ndarray):
        # calcula a distância entre as amostras e os centroids
        return self.distance(sample, self.centroids)

    def predict(self, dataset: Dataset) -> np.ndarray:
        # infere qual dos centroids está mais perto da amostra
        return np.apply_along_axis(self._get_closest_centroid, axis = 1, arr = dataset.x)

    def fit_predict(self, dataset: Dataset) -> np.ndarray:
        # infere os centroids e retorna os labels
        self.fit(dataset)
        return self.predict(dataset)


    def transform(self, dataset: Dataset) -> np.ndarray:
        # calcula as distâncias entre as amostras e os centroids
        centroids_distance = np.apply_along_axis(self._get_distances, axis = 1, arr = dataset.x)
        return centroids_distance

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        # infere os centroids e retorna as distâncias entre as amostras e os centroids
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    from si.data.dataset import Dataset
    dataset_ = Dataset.from_random(100, 5)

    k_ = 3
    kmeans = KMeans(k_)
    res = kmeans.fit_transform(dataset_)
    predictions = kmeans.predict(dataset_)
    print(res.shape)
    print(predictions.shape)