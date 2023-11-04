"""用pytorch代码实现灰度化和二值化"""

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
import cv2

# 1.导入图片
# img = cv2.imread("tiger.png")
img = plt.imread("tiger.png")

# 2.显示原图
plt.subplot(221)
plt.imshow(img)

# 3.灰度化并显示
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

# 4.二值化并显示
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
