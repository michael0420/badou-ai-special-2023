import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
直方图均衡化
'''
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
imgEqu = cv2.equalizeHist(gray)

fig = plt.figure(figsize=(7, 7))
plt.subplot(221), plt.title("Original image "), plt.axis('off')
plt.imshow(gray, cmap='gray', vmin=0, vmax=255)  # 原始图像
plt.subplot(222), plt.title("Hist-equalized image"), plt.axis('off')
plt.imshow(imgEqu, cmap='gray', vmin=0, vmax=255)  # 转换图像
histImg, bins = np.histogram(img.flatten(), 256)  # 计算原始图像直方图
plt.subplot(223, yticks=[]), plt.bar(bins[:-1], histImg)  # 原始图像直方图
plt.title("Histogram of original image"), plt.axis([0, 255, 0, np.max(histImg)])
histEqu, bins = np.histogram(imgEqu.flatten(), 256)  # 计算原始图像直方图
plt.subplot(224, yticks=[]), plt.bar(bins[:-1], histEqu)  # 转换图像直方图
plt.title("Histogram of equalized image"), plt.axis([0, 255, 0, np.max(histImg)])
plt.show()

# 彩色图像直方图均衡化
(b, g, r) = cv2.split(img)
bEqu = cv2.equalizeHist(b)
gEqu = cv2.equalizeHist(g)
rEqu = cv2.equalizeHist(r)

fig1 = plt.figure(figsize=(7, 7))
plt.subplot(341), plt.title("b image "), plt.axis('off')
plt.imshow(b, cmap='gray', vmin=0, vmax=255)
histImgB, binsB = np.histogram(b.flatten(), 256)
plt.subplot(342, yticks=[]), plt.bar(binsB[:-1], histImgB)
plt.title("Histogram of b image"), plt.axis([0, 255, 0, np.max(histImgB)])
plt.subplot(343), plt.title("bEqu image "), plt.axis('off')
plt.imshow(bEqu, cmap='gray', vmin=0, vmax=255)
histImgBEqu, binsBEqu = np.histogram(bEqu.flatten(), 256)
plt.subplot(344, yticks=[]), plt.bar(binsBEqu[:-1], histImgBEqu)
plt.title("Histogram of bEqu image"), plt.axis([0, 255, 0, np.max(histImgBEqu)])

plt.subplot(345), plt.title("g image "), plt.axis('off')
plt.imshow(g, cmap='gray', vmin=0, vmax=255)
histImgG, binsG = np.histogram(g.flatten(), 256)
plt.subplot(346, yticks=[]), plt.bar(binsG[:-1], histImgG)
plt.title("Histogram of g image"), plt.axis([0, 255, 0, np.max(histImgG)])
plt.subplot(347), plt.title("gEqu image "), plt.axis('off')
plt.imshow(gEqu, cmap='gray', vmin=0, vmax=255)
histImgGEqu, binsGEqu = np.histogram(gEqu.flatten(), 256)
plt.subplot(348, yticks=[]), plt.bar(binsGEqu[:-1], histImgGEqu)
plt.title("Histogram of gEqu image"), plt.axis([0, 255, 0, np.max(histImgGEqu)])

plt.subplot(349), plt.title("r image "), plt.axis('off')
plt.imshow(g, cmap='gray', vmin=0, vmax=255)
histImgR, binsR = np.histogram(r.flatten(), 256)
plt.subplot(3, 4, 10, yticks=[]), plt.bar(binsR[:-1], histImgR)
plt.title("Histogram of r image"), plt.axis([0, 255, 0, np.max(histImgR)])
plt.subplot(3, 4, 11), plt.title("rEqu image "), plt.axis('off')
plt.imshow(rEqu, cmap='gray', vmin=0, vmax=255)
histImgREqu, binsREqu = np.histogram(rEqu.flatten(), 256)
plt.subplot(3, 4, 12, yticks=[]), plt.bar(binsREqu[:-1], histImgREqu)
plt.title("Histogram of rEqu image"), plt.axis([0, 255, 0, np.max(histImgREqu)])

plt.show()
