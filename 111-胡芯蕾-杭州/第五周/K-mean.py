import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('lenna.png')
GRAY = True

if GRAY :
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 调整图像的形状以适应K均值算法的输入要求
    pixel_values = image.reshape((-1, 1))
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 调整图像的形状以适应K均值算法的输入要求
    pixel_values = image.reshape((-1, 3))

pixel_values = np.float32(pixel_values)

# 定义停止条件和K值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 2  # 聚类的数量

_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(image.shape)

if GRAY:
    plt.subplot(1, 2, 1)
    plt.imshow(image,cmap='gray')
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image,cmap='gray')
    plt.title('Segmented Image')
    plt.xticks([]), plt.yticks([])

    plt.show()
else:
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title('Segmented Image')
    plt.xticks([]), plt.yticks([])

    plt.show()