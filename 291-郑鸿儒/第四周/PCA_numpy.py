#!/usr/bin.env python
# encoding=utf-8
import cv2
import numpy as np


class PCANumpy(object):
    def __init__(self, n_components):
        self.k = n_components

    def fit_transform(self, X):
        center_x = X - X.mean(axis=0)
        C = np.dot(center_x.T, center_x) / X.shape[0]
        a, b = np.linalg.eig(C)
        ind = np.argsort(-1 * np.abs(a))
        U = b[:, ind[: self.k]]
        print(X, '\n', U)
        return np.dot(center_x, U)


if "__main__" == __name__:
    X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
    pca = PCANumpy(n_components=2)
    res = pca.fit_transform(X)
    print(res)
