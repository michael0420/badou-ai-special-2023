import numpy as np
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
# 计算均值
mean = np.mean(X, axis=0)
print("mean: ", format(mean))

# 中心化
centered_X = (X - mean)
print("centered_X: ", format(centered_X))

# 协方差矩阵
cov = np.cov(centered_X, rowvar=False)
print("cov: ", format(cov))

#特征值和特征向量
e_value, e_vec = np.linalg.eig(cov)
print("e_value: ", format(e_value))
print("e_vec: ", format(e_vec))

#特征值排序
idx = np.argsort(e_value)[::-1]
print("idx: ", format(idx))

#取前k个特征向量
topk_e_vec = e_vec[:, idx[:2]]
print('topk_e_vec: ', format(topk_e_vec))

# 计算降维矩阵Z
Z = np.dot(X, topk_e_vec)
print("Z: ", format(Z))
