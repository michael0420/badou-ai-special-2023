import numpy as np
import cv2
import matplotlib.pyplot as plt

# 灰度化
img = cv2.imread("lenna.png")
height, weight = img.shape[:2]  # 获取图像长宽
img_gray = np.zeros([height, weight], img.dtype)
for i in range(height):
    for j in range(weight):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)  # cv为BGR图像
img_gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cv灰度化函数

# 二值化
img_binary = img_gray.copy()
for i in range(height):
    for j in range(weight):
        if img_binary[i, j] <= 128:
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 255
 
rst,img_binary1 = cv2.threshold(img_gray1,128,255,0)

plt.subplot(221), plt.axis('off'), plt.title("gray")
plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
plt.subplot(222), plt.title("cv-gray"), plt.axis('off')
plt.imshow(img_gray1, cmap='gray', vmin=0, vmax=255)
plt.subplot(223), plt.title("binary"), plt.axis('off')
plt.imshow(img_binary, cmap='gray', vmin=0, vmax=255)
plt.subplot(224), plt.title("cv-binary"), plt.axis('off')
plt.imshow(img_binary1, cmap='gray', vmin=0, vmax=255)
plt.tight_layout()
plt.show()

