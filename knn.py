import numpy as np
from scipy import stats
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

norm_funcs = {"L2"}


class KNN:
    def __init__(self, k, metric='L2'):
        self.k = k
        self.metric = metric
        self.norm_funcs = {'L2': self.L2_dist, 'cosine': self.cosine_dist, 'manhattan': self.L1_dist}

    def fit(self, X, y):
        self.X = X
        self.y = y

    def mode(self, y):
        if len(y) == 0:
            return -1
        else:
            return stats.mode(y.flatten())[0][0]

    # Taken from : https://stackoverflow.com/questions/47736531/vectorized-matrix-manhattan-distance-in-numpy
    def L1_dist(self, X, Xtest):
        """
        Calculates the L1 distance between each of X[i] and Xtest[j] for all i and all j
        """
        X = normalize(X, axis=1)
        Xtest = normalize(Xtest, axis=1)
        return np.abs(X[:, 0, None] - Xtest[:, 0]) + np.abs(X[:, 1, None] - Xtest[:, 1])

    def L2_dist(self, X, Xtest):
        """
        Calculates the L2 distance between each of X[i] and Xtest[j] for all i and all j
        """
        X = normalize(X, axis=1)
        Xtest = normalize(Xtest, axis=1)
        return np.sum(X ** 2, axis=1)[:, None] + np.sum(Xtest ** 2, axis=1)[None] - 2 * np.dot(X, Xtest.T)

    def cosine_dist(self, X, Xtest):
        """
        Calculates the L2 distance between each of X[i] and Xtest[j] for all i and all j
        """
        X = normalize(X, axis=1)
        Xtest = normalize(Xtest, axis=1)
        z = np.dot(X, Xtest.T)
        return 1 - z

    def predict(self, Xtest):
        D = self.norm_funcs[self.metric](self.X, Xtest)

        T = D.shape[1]
        yhat = np.ones(T)

        for t in range(T):
            distances = D[:, t]
            indexes = np.argsort(distances)[:self.k]
            yhat[t] = self.mode(self.y[indexes])

        return yhat
