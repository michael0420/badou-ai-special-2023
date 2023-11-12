import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


# X 目标矩阵 K 降维后矩阵阶层
class self_PCA(object):
    def __init__(self, X, K):
        # 目标矩阵
        self.X = X
        # 降维后矩阵阶层
        self.K = K
        # 目标中心化后矩阵
        self.center_X = []
        # 目标协方差矩阵
        self.C = []
        #  目标降维转换矩阵
        self.U = []
        #  目标降维后结果矩阵
        self.res = []

        self._centerLized()
        self._cov()
        self._U()
        self._res()

    # 目标中心化函数
    def _centerLized(self):
        # 样本值的特征均值
        mean = np.array([np.mean(attr) for attr in self.X.T])
        self.center_X = self.X - mean

    # 计算目标协方差矩阵
    def _cov(self):
        # 样本集样例总数
        n_samples_num = np.shape(self.center_X)[0]
        # 中心化矩阵的协方差矩阵公式 X^T * X / n_samples_num
        self.C = np.dot(self.center_X.T, self.center_X) / (n_samples_num - 1)

    # 计算目标降维转换矩阵
    def _U(self):
        # 计算协方差矩阵C的特征值和特征向量 a 特征值 b 特征向量
        a, b = np.linalg.eig(self.C)
        # 计算特征值降序的索引序列
        ind = np.argsort(-1 * a)
        # 构建转换矩阵
        UT = [b[:, ind[index]] for index in range(self.K)]
        self.U = np.transpose(UT)

    # 计算结果
    def _res(self):
        self.res = np.dot(self.X, self.U)


# 加载数据，x表示数据集中的属性数据，y表示数据标签
X, y = load_iris(return_X_y=True)
pca = self_PCA(X, 2)
res = pca.res

# 接口调用
# pca = PCA(n_components=2)   #降到2维
# pca.fit(X)                  #训练
# res = pca.fit_transform(X)   #降维后的数据

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

# 按鸢尾花的类别将降维后的数据点保存在不同的表中
for i in range(len(res)):
    if y[i] == 0:
        red_x.append(res[i][0])
        red_y.append(res[i][1])
    elif y[i] == 1:
        blue_x.append(res[i][0])
        blue_y.append(res[i][1])
    else:
        green_x.append(res[i][0])
        green_y.append(res[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
