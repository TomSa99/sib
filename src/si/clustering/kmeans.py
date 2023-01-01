from typing import Callable
import numpy as np

from src.si.statistics.euclidean_distance import euclidean_distance
from src.si.data.dataset import Dataset


class KMeans:
    """
    It performs the k-means algorithm on the dataset.
    The k-means algorithm groups samples into groups called centroids.
    The algorithm tries to reduce the distance between the samples and the centroid

    Parameters
    ----------
    :param k: int
        Number of clusters
    :param max_iter: int
        Maximum number of iterations
    :param distance: Callable
        Distance function

    Attributes
    ----------
    centroids: np.ndarray
        Centroids
    labels: np.ndarray
        Labels
    """
    def __init__(self, k: int, max_iter: int = 1000, distance: euclidean_distance = euclidean_distance):
        # parameters
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        # attributes
        self.centroids = None
        self.labels = None

    def _init_centroids(self, dataset: Dataset):
        """
        Initializes the centroids

        :param dataset: Dataset
            Dataset object
        """
        seeds = np.random.permutation(dataset.x.shape[0])[:self.k]  # sample in each centroid
        self.centroids = dataset.x[seeds]  # seeds = sample 5 / sample 10 / sample 15...
        # print(self.centroids)

    def _get_closest_centroid(self, sample: np.ndarray) -> np.ndarray:
        """
        Returns the closest centroid

        :param sample: np.ndarray
            A sample

        :return: np.ndarray
            The closest centroid
        """
        # calculate the distance between samples and centroidss
        centroids_distance = self.distance(sample, self.centroids)
        # calculates the index of the closest centroid of the sample
        closest_centroids_index = np.argmin(centroids_distance, axis=0)
        return closest_centroids_index

    def fit(self, dataset: Dataset):
        """
        It fits the k-means clustering on the dataset
        infers the centroids by minimizing the distance between the samples and the centroids

        :param dataset: Dataset
            Dataset object

        :return: KMeans
            KMeans object
        """
        # iniciates the centroids
        self._init_centroids(dataset)

        # fitting the k-means
        convergence = False
        j = 0
        labels = np.zeros(dataset.shape()[0])
        while not convergence and j < self.max_iter:

            # get closest centroids
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.x)

            # compute new centroids
            centroids = []
            for i in range(self.k):
                centroid = np.mean(dataset.x[new_labels == i], axis=0)
                centroids.append(centroid)
            self.centroids = np.array(centroids)

            # checks if centroids have converged
            convergence = np.any(labels != new_labels)

            # update the labels
            labels = new_labels

            # increment the counter
            j += 1
        self.labels = labels
        return self

    def _get_distances(self, sample: np.ndarray):
        """
        Returns the distances between the sample and the centroids

        :param sample: np.ndarray
            A sample

        :return: np.ndarray
            The distances between each sample and the centroids
        """
        return self.distance(sample, self.centroids)

    def transform(self, dataset: Dataset):
        """
        Transforms the dataset
        Returns the distances between each samples and the centroids

        :param dataset: Dataset
            Dataset object

        :return: np.ndarray
            Transformed dataset
        """
        centroids_distance = np.apply_along_axis(self._get_distances, axis=1, arr=dataset.x)
        return centroids_distance

    def fit_transform(self, dataset: Dataset):
        """
        It fits and transforms the dataset

        :param dataset: Dataset
            Dataset object

        :return: np.ndarray
            Transformed dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

    def predict(self, dataset: Dataset):
        """
        Predicts the labels of the dataset

        :param dataset: Dataset
            Dataset object

        :return: np.ndarray
            Predicted labels
        """
        # infers which of the centroids is closest to the sample
        return np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.x)

    def fit_predict(self, dataset: Dataset):
        """
        It fits and predicts the labels of the dataset

        :param dataset: Dataset
            Dataset object

        :return: np.ndarray
            Predicted labels
        """
        self.fit(dataset)
        return self.predict(dataset)

if __name__ == "__main__":
    X = np.array([[0, 2, 0, 3],
                  [0, 1, 4, 3],
                  [0, 1, 1, 3]])
    a = KMeans(2, 10)
    dataset = Dataset(X,
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    k = 3
    kmeans = KMeans(k)
    res = kmeans.fit_transform(dataset)
    predictions = kmeans.predict(dataset)
    print(res.shape)
    print(predictions.shape)