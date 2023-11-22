

# kmean 聚簇
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna2.jpg', 0)
h, w = img.shape[:2]
data = img.reshape(h*w, 1)
data = np.float32(data)
# 聚簇类数
K = 3
# 停止条件
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
# 最大重复计算次数
N = 10
# 初始中心
flags = cv2.KMEANS_RANDOM_CENTERS
com, dst_data, center = cv2.kmeans(data, K, None, criteria, N, flags)

dst = dst_data.reshape((h, w))



plt.imshow(dst,'gray')
plt.show()


