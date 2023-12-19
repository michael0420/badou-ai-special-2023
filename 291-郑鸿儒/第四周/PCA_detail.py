#!/usr/bin/env/ python
# encoding=utf-8
import cv2
import numpy as np


class PCADetail(object):
    def __init__(self, X, information):
        """
        information: 保有信息量
        """
        self.X = X
        self.f = information
        self.centrex = self._centralized()
        # print(self.centrex)
        self.C = self._cov()
        self.U = self._u()
        self.Z = self._z()
        # print(self.U)
        # print(self.Z)

    def _centralized(self):
        mean = np.array([np.mean(attr) for attr in self.X.T])
        return self.X - mean

    # 中心化协方差矩阵 1 / n * x.T * x
    def _cov(self):
        # print(self.X.shape[0])
        return 1 / self.X.shape[0] * np.dot(self.centrex.T, self.centrex)

    # 特征向量按特征值模的大小从大到小排列
    def _u(self):
        a, b = np.linalg.eig(self.C)
        # 此处应是特征值模的大小比较，而eig函数返回的值没有取模
        ind = np.argsort(-1 * np.abs(a))
        # 从一维开始计算保有信息量
        k = 1
        for i in range(1, self.X.shape[1]):
            # 计算保有信息量
            in_percetage = np.abs(sum([a[ind[n]] for n in range(i)]) / sum(a))
            print([a[ind[n]] for n in range(i)])
            print(sum(a))
            print(in_percetage)
            # 高于预定值则使用当前值作为k值
            if in_percetage > self.f:
                k = i
        # print(ind)
        # rangNum =
        return np.transpose([b[:, ind[i]] for i in range(k)])
        # print(a)
        # print(b)
        # print(np.argsort(a * -1))

    # 样本矩阵(中心化)与特征向量矩阵相乘
    def _z(self):
        # print(self.X.shape, self.U.shape)
        Z = np.dot(self.centrex, self.U)
        print(Z)
        return Z


if "__main__" == __name__:
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    PCADetail(X, 0.9)
