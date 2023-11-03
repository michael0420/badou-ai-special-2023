import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
@author: zhangxi

图片基本处理：灰度+二值化
"""

# img = cv2.imread("lenna.png")
# print(img)
# h, w = img.shape[:2]
# img_gray = np.zeros([h, w], img.dtype)

# for i in range(h):
#     for j in range(w):
#         m = img[i, j]
#         img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
#
# print("img show gray%s"%img_gray)
# cv2.imshow("image show gray", img_gray)
#
# cv2.waitKey(0)

plt.subplot(221)
img = plt.imread("lenna.png")
# print(img)
plt.imshow(img)

# 灰度化
plt.subplot(222)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, "gray")

# print("img show gray%s"%img_gray)

plt.subplot(223)
h, w = img_gray.shape[:2]
for i in range(h):
    for j in range(w):
        if(img_gray[i,j] > 0.5):
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 1

plt.imshow(img_gray, "gray")


plt.show()