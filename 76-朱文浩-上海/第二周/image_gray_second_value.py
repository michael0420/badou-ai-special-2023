import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

'''
手动实现灰度化
'''
# 读取图片
img = cv2.imread('img.png')
# 读取图片长，宽
h, w = img.shape[:2]

img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)

print(img_gray)
cv2.imshow('gray picture ',img_gray)
cv2.waitKey(3)



'''
手动二值化
'''

# img = plt.imread('img.png')
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if(img_gray[i, j]<=0.5):
#             img_gray[i,j] = 0
#         else:
#             img_gray[i,j] = 1
#
# plt.imshow(img_gray,cmap="gray")


"""
numpy实现图像二值化
"""

img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.imshow(img_binary,cmap="gray")
plt.show()



