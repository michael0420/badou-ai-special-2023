import numpy as np

class PCA(object):
    '''用PCA求样本x的k阶降维矩阵z
    Note:输入的样本矩阵x的shape=(m, n)，m行样例，n个特征
    '''
    def __init__(self, x, k):
        '''
        :param x: 训练样本矩阵x
        :param k: x的降维矩阵的阶数，即x要特征降维成k阶
        '''
        self.x = x          #样本矩阵x
        self.k = k          #k阶降维矩阵的K值
        self.centr_x = []   #矩阵x的中心化
        self.C = []         #样本集的协方差矩阵C
        self.U = []         #样本矩阵X的降维转换矩阵
        self.Z = []         #样本矩阵X的降维矩阵Z

        self.centr_x = self._centralized()
        self.C = self._cov()
        self.U = self._u()
        self.Z = self._z()  #Z=XU

    def _centralized(self):
        '''矩阵X的中心化'''
        centrX = []         #矩阵x的中心化矩阵
        meanX = np.array([np.mean(attr) for attr in self.x.T]) #样本集求特征均值
        centrX = self.x - meanX #矩阵x的中心化
        return centrX

    def _cov(self):
        '''求矩阵X中心化后的协方差矩阵C'''
        n = np.shape(self.centr_x)[0] #样本集的样例总数
        covX = np.dot(self.centr_x.T, self.centr_x)/(n-1) #样本矩阵的协方差矩阵C
        return covX

    def _u(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        #先求X的协方差矩阵C的特征值和特征向量
        #特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        a, b = np.linalg.eig(self.C)
        #给出特征值降序的topK的索引序列
        ind = np.argsort(-a)
        #构建K阶降维的降维转换矩阵U
        UT = np.array([b[:, ind[i]] for i in range(self.k)])
        U = np.transpose(UT)
        return U

    def _z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        z = np.dot(self.centr_x, self.U)
        print("X shape:", self.x.shape)
        print("U shape:", self.U.shape)
        print("Z shape:", z.shape)
        print("样本矩阵X的降维矩阵Z:\n", z)
        return z

if __name__ == "__main__":
    '''10个样本3个特征样本集，行为样本，列为特征'''
    x = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    k = x.shape[1] - 1
    print("样本集(10行3列，10个样例，每个样例3个特征):\n", x)
    pca = PCA(x, k)
    # meanX = np.array([np.mean(attr) for attr in x.T])
    # c = x - meanX
    # cov = np.dot(c.T, c)/(x.shape[0]-1)
    # a, b = np.linalg.eig(cov)
    # ind = np.argsort(-a)
    # U = np.array([b[:, ind[i]] for i in range(k)]).T
    # Z = np.dot(x, U)
    # print(Z)

