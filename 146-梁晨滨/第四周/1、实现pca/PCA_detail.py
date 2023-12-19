# -*- coding: utf-8 -*-
import numpy as np
# from matplotlib import pyplot as plt
# import random

# 只有在nn.moudle中调用类才会自动执行forward


class PCA(object):
    def __init__(self, input, k):
        self.input_array = input
        self.k = k

        self.array_center = self.center()
        self.con = self.conxy()
        self.trans = self.array_trans()
        self.output = self.out()

    # 计算中心
    def center(self):
        array_mean = np.mean(self.input_array.T)
        array_center = self.input_array - array_mean
        return array_center

    # 计算协方差矩阵
    def conxy(self):
        total = np.shape(self.array_center)[0]
        con = np.dot(self.array_center.T, self.array_center) / total
        return con

    # 计算降维矩阵
    def array_trans(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.con)
        index = np.argsort(-1 * eigenvalues)
        transT = [eigenvectors[:, index[i]] for i in range(self.k)]
            # self.transT.append(eigenvectors[:, index[i]])
        trans = np.transpose(transT)
        return trans

    # 输入乘降维矩阵得到最终结果
    def out(self):
        output = np.dot(self.input_array, self.trans)
        print("输入数组：", self.input_array)
        print("输出数组：", output)
        print("输入数组维度：", self.input_array.shape)
        print("输出数组维度：", output.shape)
        return output





if __name__ == '__main__':
    # 设置一个输入数组，每个值在0-100之间，10行3列
    input = np.random.randint(0, 100, size=(10, 4))
    # k是输出的维度，这里设定为原始维度-2
    k = np.shape(input)[-1] - 2

    output = PCA(input, k)

