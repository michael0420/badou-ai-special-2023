import cv2
import numpy as np
import matplotlib.pyplot as plt

# 参考文档 https://opencv.apachecn.org/4.0.0/8.3.2-tutorial_py_kmeans_opencv/#3

img = cv2.cvtColor(cv2.imread("../Images/lenna.png"), cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

data_img_gray = np.float32(img_gray.reshape((img_gray.shape[0] * img_gray.shape[1], 1)))
data_img_rgb = np.float32(img.reshape((-1, 3)))

# 终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 标签
flags = cv2.KMEANS_RANDOM_CENTERS

# k-means聚类
compactness_img_gray, labels_img_gray, centers_img_gray = cv2.kmeans(data_img_gray, 4, None, criteria, 10, flags)
compactness_img_rgb, labels_img_rgb, centers_img_rgb = cv2.kmeans(data_img_rgb, 8, None, criteria, 10, flags)

# 绘图
dst_gray = labels_img_gray.reshape((img_gray.shape[0], img_gray.shape[1]))
res_rgb = np.uint8(centers_img_rgb)[labels_img_rgb.flatten()]
dst_rgb = res_rgb.reshape(img.shape)

titles = ["img", "img_gray", "k_means_img_rgb", "k_means_img_gray"]
img_list = [img, img_gray, dst_rgb, dst_gray]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img_list[i], "gray")
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()


