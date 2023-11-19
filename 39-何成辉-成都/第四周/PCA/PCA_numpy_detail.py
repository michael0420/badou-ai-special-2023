# -*- coding: utf-8 -*-

"""
@author: BraHitYQ
    1.Implement a PCA (Principal Component Analysis) class for dimensionality reduction of the input data matrix.
    2.Using PCA to Find the K-th Reduced Dimension Matrix Z of Sample Matrix X.
    1.实现一个PCA（主成分分析）类，用于对输入的数据矩阵进行降维处理
    2.使用PCA求样本矩阵X的K阶降维矩阵Z
"""
 
import numpy as np


class CPCA(object):  # 定义一个名为CPCA的类，继承自Python的内置对象类。
    '''
    用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''
    def __init__(self, X, K):
        '''
        定义类的构造函数，接收两个参数：数据矩阵X和降维后的维度K。
        :param X,训练样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        # 以下几个是对对象的一个操作，主要是初始化对象
        self.X = X       # 样本矩阵X,将输入的数据矩阵X赋值给类的实例变量self.X。——————在面向对象编程中，为了区别两个一样的参数，并且同时使用获取的参数，就需要使用如下方式进行赋值（这是最简便的方式，也是统一的标准）——此方法也称为链式调用
        self.K = K       # K阶降维矩阵的K值,将输入的降维维度K赋值给类的实例变量self.K。
        self.centrX = []  # 初始化一个空列表，用于存储中心化后的数据矩阵。
        self.C = []      # 初始化一个空列表，用于存储协方差矩阵。
        self.U = []      # 初始化一个空列表，用于存储降维转换矩阵。
        self.Z = []      # 初始化一个空列表，用于存储降维后的数据矩阵。

        # 以下是对方法（函数）的”初始化“,调用内部方法（以下这些方法需要我们在CPCA(object)方法中完成定义并使用）
        self.centrX = self._centralized()  # 调用内部方法_centralized()对数据矩阵进行中心化处理，并将结果赋值给self.centrX。
        self.C = self._cov()  # 调用内部方法_cov()计算协方差矩阵，并将结果赋值给self.C。
        self.U = self._U()  # 调用内部方法_U()计算降维转换矩阵，并将结果赋值给self.U。
        self.Z = self._Z()  # 调用内部方法_Z()计算降维后的数据矩阵，并将结果赋值给self.Z,由Z=XU求得
        
    def _centralized(self):
        '''
        矩阵X的中心化
        '''

        print('样本矩阵X:\n', self.X)
        centrX = []  # 创建一个空列表centrX，用于存储中心化后的数据矩阵。
        # 计算数据矩阵X的每列特征的均值，并将结果存储在NumPy数组mean中。这里使用了列表推导式和NumPy的mean函数。
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        # 将原始数据矩阵X的每一列减去对应的均值，得到中心化后的数据矩阵，并将结果存储在centrX中。
        centrX = self.X - mean  # 样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX
        
    def _cov(self):
        '''
        求样本矩阵X的协方差矩阵C
        '''

        # 样本集的样例总数
        # 使用NumPy库的shape函数获取样本矩阵X的形状（即行数和列数），并将行数赋值给变量ns。
        ns = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C
        # 使用NumPy库的dot函数计算样本矩阵X的转置矩阵与原矩阵的乘积，然后除以(ns - 1)，得到协方差矩阵C。
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        '''
        求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度
        '''

        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)  # 使用NumPy库的linalg.eig函数计算协方差矩阵C的特征值和特征向量，分别赋值给变量a和b,函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1*a)  # 对特征值数组a进行降序排序，并返回排序后的索引数组ind。
        # 构建K阶降维的降维转换矩阵U
        # 根据排序后的索引数组ind，从特征向量矩阵b中选取前self.K个特征向量，组成一个新的矩阵UT
        UT = [b[:,ind[i]] for i in range(self.K)]
        # 将矩阵UT进行转置，得到降维矩阵U。
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U
        
    def _Z(self):
        '''
        按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数
        '''

        # 使用NumPy库的dot函数计算矩阵X和矩阵U的点积，将结果赋值给变量Z。
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z
        
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
    # 这行代码计算了样本的特征数量减一，这是因为在PCA中，我们通常保留的主成分数量等于原始特征数量减一。
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X,K)
