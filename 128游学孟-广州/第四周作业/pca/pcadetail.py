import numpy as np

# 设置打印选项，显示小数位点后7位数
np.set_printoptions(precision=7)

#创建一个示例数据库
X = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
    ])
# 1.去均值处理
X_mean = np.mean(X,axis=0)
print("X_mean",X_mean)
X_centered = X - X_mean

#2.计算协方差矩阵,协方差矩阵 = (中心化矩阵的转置 × 中心化矩阵) / (样本数量 - 1)

#计算数据量的点
num = X.shape[0]
print("数据点的数量:")
print(num)
print()

#计算中心化后的矩阵的转置
X_centered_T = X_centered.T
print("中心化后的矩阵的转置:")
print(X_centered_T)
print()

#计算协方差矩阵
cov_matrix = np.dot(X_centered_T,X_centered)/(num - 1)

#3.特征值分解
eigenvalues,eigenvectors = np.linalg.eig(cov_matrix)

#4.特征值值排序
sorted_indexes = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indexes]
sorted_eigenvetors = eigenvectors[:,sorted_indexes]

#5.选择主成分（特征向量）
k = 2 #选择降维后的维度
selected_eigenvectors = sorted_eigenvetors[:,:k]

#6.数据投影到新的低维空间
X_new = np.dot(X_centered,selected_eigenvectors)

print("原始数据:")
print(X)
print()

print("去均值处理后的数据:")
print(X_centered)
print()

print("协方差矩阵:")
print(cov_matrix)
print()

print("特征值:")
print(eigenvalues)
print()

print("特征向量:")
print(eigenvectors)
print()

print("排序后特征值:")
print(sorted_eigenvalues)
print()

print("排序后的特征向量:")
print(sorted_eigenvetors)
print()

print("选择主成分:")
print(selected_eigenvectors)
print()

print("降维度后的数据:")
print(X_new)


