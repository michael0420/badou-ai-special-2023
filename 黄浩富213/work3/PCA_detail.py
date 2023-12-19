from sklearn.decomposition import PCA
import numpy as np

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

# 降维K
k = np.shape(X)[1] - 1
print("降维：{}".format(k))


# 中心化
# mean = np.mean(X.T)
mean = np.array([np.mean(attr) for attr in X.T])
center_X = np.subtract(X, mean)
print("原矩阵:{}".format(X))
print("中心化:{}".format(center_X))

'''求样本矩阵X的协方差矩阵cov'''
ns = np.shape(center_X)[0]
cov = np.dot(center_X.T, center_X) / (ns - 1)
print('矩阵X的协方差矩阵:{}'.format(cov))

# 求X的降维转换矩阵U
e_value, e_vector = np.linalg.eig(cov)
print('特征值:{}'.format(e_value))
print('特征向量:{}'.format(e_vector))
# 给出特征值降序的topK的索引序列
index = np.argsort(e_value)[::-1]
#构建K阶降维的降维转换矩阵U
UT = [e_vector[:, index[i]] for i in range(k)]
U = np.transpose(UT)
print('2阶降维转换矩阵U:{}'.format(U))

# 按照Z=XU求降维矩阵Z
Z = np.dot(X, U)
print('降维矩阵Z:{}'.format(Z))

