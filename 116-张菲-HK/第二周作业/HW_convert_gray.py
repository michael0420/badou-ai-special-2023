
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

# 灰度化
img = plt.imread("C:/Users/User/Desktop/Ai/八斗2023AI精品班/【2】数学&数字图像/代码/lenna.png")
img_gray = rgb2gray(img)
# generate matrix
print(img)
print(img_gray)


# show images
# plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(img)

plt.subplot(1, 3, 2)
plt.imshow(img_gray, cmap='gray')

# 二值化
img_binary = np.where(img_gray >= 0.5,1,0)

plt.subplot(1, 3, 3)
plt.imshow(img_binary, cmap='gray')
plt.show()