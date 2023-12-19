import numpy as np


class LPCA(object):
    def __init__(self, X, K):

        '''
        :param X: 训练样本矩阵X
        :param k: X的降维矩阵k
        '''

        self.X = X
        self.K = K
        self.CenterX = []  # 矩阵X去中心化
        self.C = []  # 表示协方差矩阵
        self.U = []  # 降维转换矩阵
        self.Z = []  # 降维矩阵
        self.CenterX = self._decentralization()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()



    def _decentralization(self):
        ''' 矩阵x去中心化'''
        print('样本矩阵X:\n', self.X)
        CenterX = []
        # mean = np.array([np.mean(attr) for attr in self.X.T])  # 计算样本均值（每个维度）
        mean = self.X.mean(0)
        print('样本特征均值：\n', mean)
        CenterX = self.X - mean  # 样本集去中心化
        print('样本去中心化矩阵CenterX:\n', CenterX)
        return CenterX

    def _cov(self):
        ''' 求样本矩阵X协方差矩阵C'''
        ns = np.shape(self.CenterX)[0]
        C = np.dot(self.CenterX.T, self.CenterX) / (ns - 1)
        print('样本矩阵的协方差矩阵C:\n', C)
        return C

    def _U(self):
        '''求样本X的降维矩阵U,shape=(n,k),n表示X的特征维度总数，k为降维矩阵的特征维度'''
        # 求X的协方差矩阵C的特征向量与特征值
        a, b = np.linalg.eig(self.C)  # a为特征值，b为特征向量
        print('协方差矩阵C的特征值:\n', a)
        print('协方差矩阵C的特征向量:\n', b)
        ind = np.argsort(-1 * a)  # 特征值降序的topK的索引系列
        # 构建K阶降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵u:\n' % self.K, U)
        return U

    def _Z(self):
        """ 按照Z=XU求降维矩阵Z """
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__=='__main__':
    '创建样本集 10样本4特征（维度），行为样本，4为特征维度'
    X = np.array([[21, 36, 17, 19],
                  [51, 6, 26, 12],
                  [37, 14, 36, 61],
                  [2, 89, 41, 70],
                  [33, 59, 10, 20],
                  [52, 21, 89, 27],
                  [9, 46, 29, 38],
                  [9, 2, 39, 66],
                  [81, 29, 40, 128],
                  [33, 56, 6, 96]])

    K = X.shape[1] - 1  # shape[x](0) 表示行数   shape[x](1)表示列数
    print('10样本（10样本，每个样本4个特征）X:\n', X)
    pca = LPCA(X, K)
