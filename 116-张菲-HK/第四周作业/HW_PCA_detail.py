import numpy as np
from sklearn.datasets import load_iris

# 构造函数，初始化数值
class CPCA(object):
    def __init__(self, X, K):
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    # 矩阵中心化，减去均值
    def _centralized(self):
        print("样本矩阵X：\n", self.X)
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 这是一个列表推导式，它用来遍历 self.X 的转置（.T 表示转置）的
        # 每一列，并计算每列的平均值。attr: 这是列表推导式中的迭代变量，它代表 self.X.T 中的每一列。
        print("样本集的特征均值：\n", mean)
        centrX = self.X - mean
        print("样本矩阵X的中心化矩阵：\n", centrX)
        return centrX

    # 求协方差矩阵
    def _cov(self):
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print("样本矩阵的协方差矩阵C:\n", C)
        return C

    # 求降维转换矩阵U
    def _U(self):
        a, b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        ind = np.argsort(-1 * a)
        UT = [b[:, ind[i]] for i in range(self.K)]
        # ind: 这是一个存储特征值索引的NumPy数组，它是根据特征值的大小（降序排列）生成的。在这个列表推导式中，我们将使用这些索引来选择特征向量。
        # ind[i]: 这是列表推导式中的迭代变量，它代表了 ind 中的每个索引。
        # b[:, ind[i]]: 这部分表示从特征矩阵 b 中选择第 ind[i] 列，即选择了根据特征值降序排列的第 i 个最重要的特征向量。
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    # 求降维矩阵Z， Z = XU
    def _Z(self):
        Z = np.dot(self.X, self.U)
        print("X shape:", np.shape(self.X))
        print("U shape:", np.shape(self.U))
        print("Z shape:", np.shape(Z))
        print("样本矩阵X的降维矩阵Z:\n", Z)
        return Z
if __name__=='__main__':
    # 使用load_iris函数加载数据集
    iris = load_iris()
    K = iris.data.shape[1]-2

    print('iris数据集(150行4列，150个样例，每个样例4个特征):\n', iris)
    print(iris.data.shape)
    pca = CPCA(iris.data, K)


