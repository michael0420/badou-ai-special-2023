"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""

import numpy as np


class CPCA(object):
    def __init__(self, X, K):
        '''
        :param X:训练样本矩阵
        :param K:X的降维矩阵的阶数
        '''
        self.X = X
        self.K = K
        self.centerX = []
        self.C = []  # 样本集X的协方差矩阵
        self.U = []  # 样本集X的降维转换矩阵
        self.Z = []  # 样本X的降维矩阵

        self.centerX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):
        mean = np.array([np.mean(avg) for avg in X.T])
        centerX = self.X - mean
        print("去中心化后的数据:\n", centerX)
        return centerX

    def _cov(self):
        n = np.shape(self.centerX)[0]
        C = np.dot(self.centerX.T, self.centerX) / (n - 1)
        print("协方差矩阵C:\n", C)
        return C

    def _U(self):
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)  # 特征值负值给a，特征向量赋值给b
        print("样本的协方差矩阵的特征值：\n", a)
        print("样本的协方差矩阵的特征向量：\n", b)
        # 给出特征值降序的topK的索引序列
        sort_index_arr = np.argsort(-1 * a)
        # print("ind = ", ind) 拿到排序好的特征值最大的索引下标（特征最符合的排序），
        # 因为其下标对应着协方差特征矩阵最符合特征值的特征向量下标
        # for i in range(self.K):
        #     print("i = %d \n" % i , b[:, ind[i]])
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, sort_index_arr[i]] for i in range(self.K)]
        U = np.transpose(UT)  # transpose 函数接受一个矩阵作为输入，并返回输入矩阵的转置
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), m是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.centerX, self.U)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


X = np.array([[10, 15, 29],  # 一行一个样本，一列一个特征
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])

pca = CPCA(X, 2)


# import sklearn.decomposition as dp
#
# pca=dp.PCA(n_components=2) #加载pca算法，设置降维后主成分数目为2
# reduced_x=pca.fit_transform(X) #对原始数据进行降维，保存在reduced_x中
# print("sklearn \n", reduced_x)