import numpy as np


class KNNClassifier:

    def __init__(self, k=3, metric='euclidean'):

        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):

        # learning = remembering or placing points on multidimensional space
        self.X_train = X_train.values
        self.y_train = y_train.values

        return self

    def predict(self, X_test):

        # predicting = comparing points in multidimensional space with each other and taking closest k points
        X_test = X_test.values

        distances = self._calculate_dist(X_test)
        y_predicted = np.zeros(X_test.shape[0], dtype=self.y_train.dtype)

        for i in range(X_test.shape[0]):

            nearest_neighbors = np.argsort(distances[i])[:self.k]
            print(nearest_neighbors)
            nearest_labels = self.y_train[nearest_neighbors]

            labels_sum, value = np.unique(nearest_labels, return_counts=True)
            print(labels_sum, value)
            y_predicted[i] = labels_sum[np.argmax(value)]

        return y_predicted

    def _calculate_dist(self, X_test):

        if self.metric == 'euclidean':
            # sum of all features values for new points squared
            data_test = np.sum(X_test**2, axis=1, keepdims=True)
            # sum of all features values for old points squared
            data_train = np.sum(self.X_train**2, axis=1, keepdims=True)
            # for each distance ( new point * number of already existing points) scalar products of points
            # each row is for one new point and each column is disctance between new point and next old point
            scalars = -2 * np.dot(X_test, self.X_train.T)
            # x^2+y^2-2xy
            sum_factors = data_test + data_train.reshape(1, -1) + scalars
            # maximum to avoid numerical mistakes
            distances = np.sqrt(np.maximum(0, sum_factors))
            print(distances)

            return distances
        else:
            raise ValueError(f"Please use euclidean metric instead of: {self.metric}")

    def __call__(self, X_test, X_train=None, y_train=None):

        if X_train is not None and y_train is not None:
            self.fit(X_train, y_train)

        return self.predict(X_test)
