import cv2
import numpy as np
import matplotlib.pyplot as plt

# K-Means算法
# 读取图像
image = cv2.imread('lenna.png')

# 将图像从BGR转换为RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像数据重塑为二维数组
pixels = image_rgb.reshape((-1, 3))

# 将数据转换为float32类型
pixels = np.float32(pixels)

# 定义K-Means参数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 2  # 设置K值为2

compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# 将每个像素标记为0或1
segmented_image = np.uint8(centers[labels.flatten()])
# 将分割后的图像重塑为原始形状
segmented_image = segmented_image.reshape(image_rgb.shape)

# 显示原始图像和分割后的图像
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title(f'K-Means Segmentation (K={k})')

plt.show()
