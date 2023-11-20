import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, x):

        center = x - x.mean(axis=0)
        cov = np.dot(center.T, center) / (x.shape[0] - 1)
        eig_vals, eig_vectors = np.linalg.eig(cov)
        index = np.argsort(-1 * np.abs(eig_vals))
        need_index = index[:self.n_components]
        components = eig_vectors[:, need_index]
        return np.dot(x, components)


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
    pca = PCA(n_components=2)
    newX = pca.fit_transform(x)
    print(newX)
