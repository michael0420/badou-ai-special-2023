import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h,w= img_gray.shape[:2]
#灰度直方图
plt.figure()
plt.hist(img_gray.ravel(), 256)
plt.show()
#


#彩色直方图
# image = cv2.imread('lenna.png')
# cv2.imshow("Original",image)
# #cv2.waitKey(0)
#
# chans = cv2.split(image)
# colors = ("b","g","r")
# plt.figure()
# plt.title("Flattened Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
#
# for (chan,color) in zip(chans,colors):
# hist = cv2.calcHist([chan],[0],None,[256],[0,256])
# plt.plot(hist,color = color)
# plt.xlim([0,256])
# plt.show()
#直方图均衡化
#方法一
hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
for i in range(1,256):
    hist[i] = hist[i] + hist[i-1]
im = np.zeros((h,w),dtype=np.uint8)
for i in range(h):
    for j in range(w):
        p = img_gray[i,j]
        im[i,j] = int(hist[p]*256/h/w+0.5)
#方法二
dst = cv2.equalizeHist(img_gray)
print(im-dst)
print(dst)
cv2.imshow("Histogram Equalization", np.hstack([img_gray,im,dst]))
cv2.waitKey(0)

