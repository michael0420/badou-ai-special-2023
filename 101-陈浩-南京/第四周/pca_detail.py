import numpy as np

#使用PCA算法求出样本矩阵的K阶降维矩阵Z

class PCAClass(object):
    '''
    使用PCA算法求出样本矩阵的K阶降维矩阵Z
    样本矩阵X 样式 shape = (m,n)， m行样例，n个特征
    '''
    def __init__(self, X, K):
        '''
        :param X: 待训练的样本矩阵X
        :param K: X的降维矩阵的阶数，即X要特征降维成K阶
        '''
        self.X = X
        self.K = K
        self.centerX = [] #矩阵X的中心化
        self.Cov = [] #样本集的协方差矩阵Cov
        self.Util = [] #样本矩阵X的降维转换矩阵
        self.Z = [] #样本矩阵X的降维矩阵Z
        self.centerX = self._centerlized()
        self.Cov = self._cov()
        self.Util = self._U()
        self.Z = self._Z()


    def _centerlized(self):
        '''将矩阵X进行中心化'''
        print('样本矩阵X：\n',self.X)
        centerX = []
        #样本集的特征均值
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print('样本集的特征均值：\n',mean)
        centerX = self.X - mean #样本集中心化
        print('样本矩阵X的中心化centrX:\n', centerX)
        return  centerX

    def _cov(self):
        '''求样本矩阵X的协方差矩阵Cov'''
        #样本集的样例总数
        ns = np.shape(self.centerX)[0]
        #样本矩阵的协方差矩阵Cov
        Cov = np.dot(self.centerX.T, self.centerX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C：\n',Cov)
        return Cov

    def _U(self):
        '''求X的降维转换矩阵U，shape=(n,k)，n是X的特征维度总数，k是降维矩阵的特征维度'''
        #先求X的协方差矩阵C的特征值和特征向量
        # 特征值赋值给a，对应特征向量赋值给b。
        # 函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        a,b = np.linalg.eig(self.Cov)
        print('样本集的协方差矩阵C的特征值：\n',a)
        print('样本集的协方差矩阵C的特征向量：\n',b)
        #给出特征值降序的topK的索引序列
        ind = np.argsort(-1*a)
        #构建K阶降维的降维转换矩阵U
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n'%self.K,U)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z，shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.Util)
        print('X shape:',np.shape(self.X))
        print('U shape:',np.shape(self.Util))
        print('Z shape:',np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n',Z)
        return Z

if __name__ == '__main__':
    '10个样本，3个特征的样本集，行位样例，列为特征维度'
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
    print('样本集（10行3列，10个样例，每个样例3个特征）：\n',X)
    pca = PCAClass(X,K)












