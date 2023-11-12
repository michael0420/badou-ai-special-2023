import numpy as np
from sklearn.datasets import load_iris


class CPCA(object):
    def __init__(self, X, K):
        # X的K阶降维
        self.X = X        # 目标矩阵
        self.K = K        # 降维后矩阵阶层
        self.centrX = []  # 目标中心化后矩阵
        self.C = []       # 目标协方差矩阵
        self.U = []       # 目标降维转换矩阵
        self.Z = []       # 目标降维后结果矩阵

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):

        # 样本值的特征均值
        mean = np.array([np.mean(attr) for attr in self.X.T])
        # 这是一个列表推导式，它用来遍历 self.X 的转置（.T 表示转置）的
        # 每一列，并计算每列的平均值。attr: 这是列表推导式中的迭代变量，它代表 self.X.T 中的每一列。

        # 样本集的中心化
        centrX = self.X - mean
        print("样本矩阵X的中心化矩阵：\n", centrX)

        return centrX

    def _cov(self):
        # 求协方差矩阵
        ns = np.shape(self.centrX)[0]  # 样本集的样例总数

        # 中心化矩阵的协方差矩阵公式 X^T * X / n_samples_num
        c = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print("样本矩阵的协方差矩阵C:\n", c)

        return c

    def _U(self):
        # 求降维转换矩阵U
        a, b = np.linalg.eig(self.C) # 计算协方差矩阵C的特征值和特征向量 a 特征值 b 特征向量
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)

        # 计算特征值降序的索引值
        ind = np.argsort(-1 * a)

        # 构建转换矩阵
        ut = [b[:, ind[i]] for i in range(self.K)]
        # ind: 这是一个存储特征值索引的NumPy数组，它是根据特征值的大小（降序排列）生成的。在这个列表推导式中，我们将使用这些索引来选择特征向量。
        # ind[i]: 这是列表推导式中的迭代变量，它代表了 ind 中的每个索引。
        # b[:, ind[i]]: 这部分表示从特征矩阵 b 中选择第 ind[i] 列，即选择了根据特征值降序排列的第 i 个最重要的特征向量。

        u = np.transpose(ut)
        print('%d阶降维转换矩阵U:\n' % self.K, u)

        return u

    def _Z(self):
        # 求降维矩阵Z， Z = XU
        z = np.dot(self.X, self.U)
        print("X shape:", np.shape(self.X))
        print("U shape:", np.shape(self.U))
        print("Z shape:", np.shape(z))
        print("样本矩阵X的降维矩阵Z:\n", z)

        return z


if __name__ == '__main__':
    # 使用load_iris函数加载数据集
    iris = load_iris()
    K = iris.data.shape[1]-2

    print('iris数据集(150行4列，150个样例，每个样例4个特征):\n', iris)
    print(iris.data.shape)
    pca = CPCA(iris.data, K)
