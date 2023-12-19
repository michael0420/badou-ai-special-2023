#实现pca(detail+调用接口)

#PCA（Principal Component Analysis）是一种常用的降维技术，用于将高维数据映射到低维空间。

import cv2
import numpy as np

# 生成示例数据
data = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]], dtype=np.float32)

# 计算PCA
mean, eigenvectors = cv2.PCACompute(data, mean=None)

# 转换数据到主成分空间
reduced_data = cv2.PCAProject(data, mean, eigenvectors)

# 查看降维后的数据
print(reduced_data)