import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 读取Lenna灰度图
lenna_gray = cv2.imread("C:/Users/15082/Desktop/lenna.png", cv2.IMREAD_GRAYSCALE)

# 将图像转换为一维数组
data = lenna_gray.flatten().reshape((-1, 1))

# 指定聚类数目
num_clusters = 5

# 使用K均值聚类
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(data)

# 获取每个像素的标签
labels = kmeans.labels_

# 将标签重新整形回图像形状
segmented_image = labels.reshape(lenna_gray.shape)

# 显示原始图像和聚类结果
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(lenna_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image (K-Means)')
plt.imshow(segmented_image, cmap='viridis')
plt.axis('off')

plt.show()
