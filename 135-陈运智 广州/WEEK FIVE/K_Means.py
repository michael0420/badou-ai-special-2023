import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread("lenna.png")

# 将图像转换为 NumPy 数组
img_array = img.reshape((-1, 3))
img_array = np.float32(img_array)

# 聚类数量
k = 2

# 使用 K-means 进行聚类
kmeans = KMeans(n_clusters=k)
kmeans.fit(img_array)

# 获取聚类结果
labels = kmeans.labels_
centers = kmeans.cluster_centers_
centers = np.uint8(centers)
print(labels)
# 将聚类结果映射回图像
segmented_img = centers[labels].reshape(img.shape)

# 保存聚类结果为图像文件
cv2.imwrite("segmented_image.jpg", segmented_img)

