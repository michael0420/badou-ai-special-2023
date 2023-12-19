import numpy as np
import tkinter


class CPCA(object):                                         # 用PCA求样本矩阵X的K阶降维矩阵Z
    def __init__(self, X, K):                               # Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征

        self.X = X                                          # 样本矩阵X
        self.K = K                                          # K阶降维矩阵的K值
        self.centrX = []                                    # 矩阵X的中心化
        self.C = []                                         # 样本集的协方差矩阵C
        self.U = []                                         # 样本矩阵X的降维转换矩阵
        self.Z = []                                         # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()                                  # Z=XU求得

    def _centralized(self):                                 # 矩阵中心化
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 先求一个特征均值
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean                              #然后把每一个都减去均值得到的样本集
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):
        ns = np.shape(self.centrX)[0]  # 中心化后样本集合
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)  # 样本集合的协方差矩阵
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        a, b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值:\n', a)               # 特征值赋值给a，对应特征向量赋值给b
        print('样本集的协方差矩阵C的特征向量:\n', b)
        ind = np.argsort(-1 * a)
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':                                  # 主体函数
    X = np.array([[10, 15, 29],                             # 每一个随机挑选的样本都带有3个特征
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1                                  # -1：将原有3个特征的三维函数降低一维变成二维函数
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X, K)                                        # 调用PCA函数

    red_x, red_y = [], []                                   # 将图像可视化
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(pca.Z)):                             # 按鸢尾花的类别将降维后的数据点保存在不同的表中
        if y[i] == 0:
            red_x.append(pca.Z[i][0])
            red_y.append(pca.Z[i][1])
        elif y[i] == 1:
            blue_x.append(pca.Z[i][0])
            blue_y.append(pca.Z[i][1])
        else:
            green_x.append(pca.Z[i][0])
            green_y.append(pca.Z[i][1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()

'''
sklearn方式
pca = dp.PCA(n_components=2)  # 加载pca算法，设置降维后主成分数目为2
reduced_x = pca.fit_transform(X)  # 对原始数据进行降维，保存在reduced_x中
print('样本矩阵X的降维矩阵reduced_x:\n', reduced_x)
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_x)):  # 按鸢尾花的类别将降维后的数据点保存在不同的表中
   if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
'''