# -*- coding=utf-8 -*-

"""
PCA算法手写(部分)
"""
import numpy as np

class pca_detail(object):

    def __init__(self,X,k):
        self.X = X
        self.k = k 

        self.C = []
        self.Z = []

        self.C = self._C()
        self.Z = self._Z()

    #计算协方差矩阵  
    def _C(self):
        n = self.X.shape[0]
        print('原矩阵X为：\n',self.X)
        X = self.X - np.mean(self.X,axis=0)
        print('中心化矩阵centerX为：\n',X)
        C = np.dot(X.T,X)/(n-1)
        print('协方差矩阵C为：\n',C)
        return C
    
    #计算降维矩阵U和最终矩阵Z
    def _Z(self):
        a,b  = np.linalg.eig(self.C)
        #特征值进行排序的序列号
        ind = np.argsort(-1*a)
        U = b[:,ind[:self.k]]
        print('降维矩阵U为：\n',U)
        Z = np.dot(X,U)
        print('最终矩阵Z为：\n',Z)
        return Z

    
    
X = np.array([[10, 15, 29],
            [15, 46, 13],
            [23, 21, 30],
            [11, 9,  35],
            [42, 45, 11],
            [9,  48, 5],
            [11, 21, 14],
            [8,  5,  15],
            [11, 12, 21],
            [21, 20, 25] ])
k = X.shape[1]-1
pca_detail(X,k)
