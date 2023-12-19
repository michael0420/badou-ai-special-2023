import numpy as np


class PCA(object):

    def __init__(self, X, K):
        # X为训练样本矩阵
        # K为降为K阶矩阵，即X要特征降维成K阶
        self.X = X
        self.K = K
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self.zxh()
        self.C = self.xfc
        self.U = self.zhjz
        self.Z = self.jwjz

    def zxh(self):
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值，每个特征求均值
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def xfc(self):
        ns = np.shape(self.centrX)[0]  # 样本集的样本个数
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def zhjz(self):
        a, b = np.linalg.eig(self.C)  # a为协方差矩阵C的特征值，b为特征向量
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        ind = np.argsort(-1 * a)
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def jwjz(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
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
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X, K)
