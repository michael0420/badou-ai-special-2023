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

#  PCA降维 取出显著特征进行分析

# 1、数据中心化
mean_data = np.mean(X, axis=0)
new_data = X - mean_data
# 计算协方差
cov_data = np.cov(new_data, rowvar=0)
# 计算特征值和特征向量,并取出topN
eig_value, eig_vec = np.linalg.eig(np.mat(cov_data))
eig_value_index = np.argsort(-eig_value)
top_eig_vec = eig_vec[:, eig_value_index[:2]]
# 原始数据对特征向量进行映射
dst_data = new_data * top_eig_vec
print('原始数据*top特征向量：', dst_data)