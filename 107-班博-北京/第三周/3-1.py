import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

def grayscale(image):
    height, width = image.shape[:2]
    img_gray = np.zeros((height, width), dtype=np.uint8)  # Declare img_gray as a numpy array
    for i in range(height):
        for j in range(width):
            m = image[i, j]
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # BGR
    return img_gray


# 获取灰度图像
img = cv2.imread("lenna.png", 1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = grayscale(img)
# cv2.imshow("image_gray", gray)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)


# # 彩色图像直方图均衡化
# img = cv2.imread("lenna.png", 1)
# # cv2.imshow("src", img)
#
# # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 合并每一个通道
# result = cv2.merge((bH, gH, rH))
# cv2.imshow("dst_rgb", result)
#
# cv2.waitKey(0)


# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

'''
images：输入图像，可以是单通道或多通道图像，数据类型为uint8或float32。
channels：指定要计算直方图的通道列表。对于灰度图像，通道值为[0]；对于彩色图像，可以指定通道值为[0]、[1]或[2]，分别对应蓝色、绿色和红色通道。
mask：可选参数，用于指定感兴趣区域。如果不需要，可以设置为None。
histSize：指定直方图的大小，即灰度级别的数量。
ranges：指定像素值的范围，通常设置为[0, 256]。
'''

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))  #拼接
cv2.waitKey(0)
