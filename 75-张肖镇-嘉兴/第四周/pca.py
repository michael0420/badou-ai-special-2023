# 步骤
# 求协方差矩阵
# 对协方差矩阵求特征向量和特征值， 特征值组成特征空间


import numpy as np


def my_pca(x, k): # x矩阵，取k维
    m = np.shape(x)[0] # 样本个数
    avg = np.average(x, axis=0)
    x0 = x - avg
    cov = np.matmul(x0.T, x0) / (m-1)# 协方差矩阵
    a,b = np.linalg.eig(cov) # a是特征值，b是特征向量 , 这里a的shape是(m) , b的shape是(维度数,样本个数)

    index = np.argsort(-1*a)[:k]
    p = b[:,index]
    return np.matmul(x,p)  # 所以这里是x矩阵乘p


if __name__ == "__main__":
    input = np.random.normal(0,1,(3,10))
    # input = np.arange(20 ,dtype=np.float64)
    input = input.reshape((-1,10)) # 10个样本 ， 维度 10
    output=my_pca(input , 9)
    print(output.shape)
