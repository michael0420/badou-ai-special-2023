import numpy as np

class PCA_detail(object):
    def __init__(self,X,k):
        '''
        X 为样本空间
        k 取前k大特征值对应的特征向量
        '''
        self.X = X
        self.k = k
        self.center = []
        self.C = []
        self.W = []
        self.Z = []

        self._center()
        self._cov()
        self._W()
        self._ans()

    # 中心化
    def _center(self):
        mean  = np.array([np.mean(i) for i in self.X.T])
        print('特征均值:\n',mean)
        self.center = self.X - mean
        print('中心化矩阵:\n',self.center)

    #协方差矩阵
    def _cov(self):
        n = np.shape(self.center)[0] # 样本个数
        s =self.center
        self.C = np.dot(s.T,s)/(n-1)

    #协方差矩阵的特征值及前k特征向量
    def _W(self):
        a, b = np.linalg.eig(self.C)
        inx = np.argsort(-1 * a) # 特征值排序
        self.W = np.array([b[:,inx[i]] for i in range(self.k)]).T

    def _ans(self):
        self.Z = np.dot(self.X, self.W)
        print('结果矩阵：',self.Z)


if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA_detail(X,K)