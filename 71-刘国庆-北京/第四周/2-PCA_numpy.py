# 导入NumPy库，用于处理数学计算，特别是多维数组和矩阵运算。
import numpy as np


# 定义主成分分析（PCA）类
class PCA:
    # 初始化方法，接受主成分个数作为参数
    def __init__(self, n_components):
        # 设置保留的主成分个数
        self.U = None
        self.C = None
        self.X = None
        self.n_feature = None
        self.n_components = n_components

    # 定义PCA降维操作的方法，接受输入数据X作为参数
    def fit_transform(self, X):
        # 计算输入数据的特征数
        self.n_feature = X.shape[1]
        # 数据中心化，计算每个特征的均值，然后将每个特征减去相应的均值
        self.X = X - X.mean(axis=0)
        # 计算数据的协方差矩阵
        self.C = np.dot(X.T, X) / X.shape[0]
        # 计算协方差矩阵的特征值和特征向量
        a, b = np.linalg.eig(self.C)
        # 获取特征值降序排列的序号
        ind = np.argsort(-1 * a)
        # 选择对应的特征向量作为降维矩阵
        self.U = b[:, ind[:self.n_components]]
        # 对输入数据进行降维操作，得到降维后的数据
        return np.dot(self.X, self.U)


# 创建PCA类的实例，指定保留2个主成分
pca = PCA(n_components=2)
# 定义输入数据X，包含6个样本，每个样本有4个特征
X = np.array([[-1, 2, 66, -1],
              [-2, 6, 58, -1],
              [-3, 8, 45, -2],
              [1, 9, 36, 1],
              [2, 10, 62, 1],
              [3, 5, 83, 2]])
# 调用PCA类的fit_transform方法，进行降维操作
newX = pca.fit_transform(X)
# 打印降维后的数据
print(newX)
