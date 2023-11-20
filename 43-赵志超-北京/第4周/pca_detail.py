import numpy as np


class PCA(object):
    def __init__(self, x, k):
        self.x = x
        self.k = k
        self.centerX = self._centralized()
        self.cov = self._cov()
        self.u = self._u()
        self.z = self._z()

    def _centralized(self):
        mean = np.array([np.mean(row) for row in self.x.T])
        return self.x - mean

    def _cov(self):
        ns = np.shape(self.centerX)[0]
        return np.dot(self.centerX.T, self.centerX) / (ns - 1)

    def _u(self):
        a, b = np.linalg.eig(self.cov)
        index = np.argsort(-1 * np.abs(a))
        ut = [b[:, index[i]] for i in range(self.k)]
        return np.transpose(ut)

    def _z(self):
        return np.dot(self.x, self.u)


if __name__ == '__main__':
    x = np.array([
        [10, 15, 29],
        [15, 46, 13],
        [23, 21, 30],
        [11, 9, 35],
        [42, 45, 11],
        [9, 48, 5],
        [11, 21, 14],
        [8, 5, 15],
        [11, 12, 21],
        [21, 20, 25]
    ])
    k = np.shape(x)[1] - 1
    pca = PCA(x, k)
    print('X:', pca.x)
    print('U:', pca.u)
    print("Z:", pca.z)
