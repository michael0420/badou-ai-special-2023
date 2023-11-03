import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

# 1.原图
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

# 2.灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

# 3.二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')



# 5.灰度化
img_cv = cv2.imread("lenna.png")
#获取图片的high和wide
h,w = img_cv.shape[:2]
#创建一张和当前图片大小一样的单通道图片
img_gray_cv = np.zeros([h,w],img_cv.dtype)
#将BGR坐标转化成gray坐标
for i in range(h):
    for j in range(w):
        m = img_cv[i,j]
        img_gray_cv[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
cv2.imshow("image show gray cv",img_gray_cv)

# 6.二值化
# rows, cols = img_gray_cv.shape[:2]
# img_binary_cv = np.zeros([rows,cols],img_gray_cv.dtype)
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray_cv[i, j] <= 0.5):
#             img_binary_cv[i, j] = 0
#         else:
#             img_binary_cv[i, j] = 1
# cv2.imshow("image show binary cv",img_binary_cv)

plt.show()
cv2.waitKey()
