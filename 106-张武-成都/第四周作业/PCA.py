import numpy as np


class PCA(object):

    def __init__(self, X, K):
        # 样本矩阵X
        self.X = X
        # 降维矩阵维度K
        self.K = K
        # 中心化矩阵
        self.centerX = []
        # 协方差矩阵C
        self.C = []
        # 降维转换矩阵U
        self.U = []
        # 主成分分析后的矩阵
        self.Z = []

    def calc(self):
        print('原始矩阵为：\n', self.X)
        print('降维K: ', self.K)
        # 计算中心化矩阵X
        self.centerX = self.calc_center_matrix()
        # 求协方差矩阵C
        self.C = self.calc_covariance_matrix()
        # 求降维转换矩阵
        self.U = self.calc_u()
        # 求降维矩阵
        self.Z = self.calc_z()
        print(self.Z)
        return self.Z

    def calc_center_matrix(self):
        # 对每一维度先计算均值得到样本集特征均值
        for i in self.X.T:
            print(i)

        means = np.array([np.mean(i) for i in self.X.T])
        print('样本集特征均值：\n', means)
        # 样本集减均值
        centerX = self.X - means
        print('中心化后的矩阵：')
        print(centerX)

        return centerX

    def calc_covariance_matrix(self):
        # 中心化后协方差矩阵公式,这里应该是中心化后的矩阵参与计算
        # 中心化后的矩阵做对称矩阵 / 样本集数量
        # np.dot() 向量点乘和矩阵乘法
        C = np.dot(self.centerX.T, self.centerX) / (self.centerX.shape[0] - 1)
        print('协方差矩阵：')
        print(C)
        print(np.dot(self.centerX.T, self.centerX))
        print(np.matmul(self.centerX.T, self.centerX))
        print(self.centerX.T.shape, self.centerX.shape)
        return C

    def calc_u(self):
        # 求协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)
        print('特征值:', a)
        print('特征向量:', b)
        print(self.X.ndim)
        # 计算降维转换矩阵
        # 特征值降序排序，取前k个维度,返回索引
        idx = np.argsort(-a)
        print('排序后索引', idx)
        # 特征向量矩阵,
        UT = [b[:, idx[i]] for i in range(self.K)]
        print('%d,特征向量矩阵' % self.K, UT)

        U = np.transpose(UT)
        print('特征向量矩阵', U)
        return U

    def calc_z(self):
        # 原始矩阵*特征向量矩阵
        Z = np.dot(self.X, self.U)

        print('样本矩阵X:', self.X.shape)
        print('降维转换矩阵U:', self.U.shape)
        print('降维矩阵Z:', Z.shape)
        return Z


X = np.array([
    [10, 15, 29, 33],
    [15, 46, 13, 66],
    [23, 21, 30, 44],
    [11, 9, 35, 66],
    [42, 45, 11, 22],
    [9, 48, 5, 17],
    [11, 21, 14, 55],
    [8, 5, 15, 88],
    [11, 12, 21, 99],
    [21, 20, 25, 25]
])
pca = PCA(X, 2)
pca.calc()

# numpy使用方式

