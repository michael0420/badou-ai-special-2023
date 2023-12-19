import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris
import numpy as np


# detail


class CPCA(object):
    def __init__(self, X, K):
        self.X = X  # 样本矩阵
        self.K = K  # 目标阶数
        self.centerX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):  # 样本中心化
        mean = np.mean(self.X, axis=0)
        center_X = self.X - mean
        self.mean = mean
        return center_X

    def _cov(self):  # 求解样本的协方差矩阵
        ns = self.X.shape[0]
        C = np.dot(self.centerX.T, self.centerX) / (ns - 1)
        return C

    def _U(self):  # 求解降维矩阵
        a, b = np.linalg.eig(self.C)  # a: 特征值 b:特征向量
        ind = np.argsort(-1 * a)
        U = np.transpose([b[:, ind[i]] for i in range(self.K)])
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        return Z


def visualize(reduced_x, Y):
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(reduced_x)):
        if Y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif Y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
    return red_x, red_y, blue_x, blue_y, green_x, green_y

if __name__ == '__main__':
    # 数据导入
    X, Y = load_iris(return_X_y=True)  # X:数据，Y:数据对应的标签

    # detail
    K = 2
    pca_d = CPCA(X, K)
    # red_x, red_y, blue_x, blue_y, green_x, green_y =
    print('样本矩阵X的降维矩阵Z:\n', pca_d.Z)
    red_x_d, red_y_d, blue_x_d, blue_y_d, green_x_d, green_y_d = visualize(pca_d.Z, Y)



    # 调用接口
    pca = dp.PCA(n_components=2)
    reduced_x = pca.fit_transform(X)
    red_x, red_y, blue_x, blue_y, green_x, green_y = visualize(reduced_x, Y)

    fig = plt.figure()
    plt.subplot(211), plt.title("PCA"), plt.axis('off')
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.axhline(0, color='black')  # 水平坐标轴
    plt.axvline(0, color='black')  # 垂直坐标轴
    plt.subplot(212), plt.title("PCA-detail"), plt.axis('off')
    plt.scatter(red_x_d, red_y_d, c='r', marker='x')
    plt.scatter(blue_x_d, blue_y_d, c='b', marker='D')
    plt.scatter(green_x_d, green_y_d, c='g', marker='.')
    plt.axhline(0, color='black')  # 水平坐标轴
    plt.axvline(0, color='black')  # 垂直坐标轴
    plt.show()
