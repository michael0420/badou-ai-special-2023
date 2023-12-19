import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('lenna.png', 0)
rows, cols = img.shape[:]
data = img.reshape((rows * cols, 1))
print(data.shape)
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

dst = labels.reshape((img.shape[0], img.shape[1]))

titles = [u'原始图像', u'聚类图像']
plt.rcParams['font.sans-serif'] = ['SimHei']
images = [img, dst]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
