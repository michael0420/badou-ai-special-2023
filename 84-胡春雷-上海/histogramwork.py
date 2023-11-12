import numpy as np
#Histogram:直方图
import cv2

from matplotlib import pyplot as plt

# img=cv2.imread("lenna.png")
# cv2.imshow('img',img)
# #key=cv2.waitKey(1000)
#
#
# plt.figure()
#
# plt.title("Flattened Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# hist=cv2.calcHist([img],[0],None,[256],[0,256])
# plt.plot(hist)
# plt.xlim([0,256])
# plt.show()
#print(img)
#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#print(img)
#
#
# plt.figure()
# plt.hist(gray.ravel(),256,facecolor='black')
# plt.show()




# #print(hist)
# cv2.imshow("gray",gray)
# key = cv2.waitKey(0)


# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# 彩色图像均衡化,分解通道也是按照bgr
(b, g, r) = cv2.split(img)
b = cv2.equalizeHist(b)
g = cv2.equalizeHist(g)
r = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((b, g, r))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)





#print(hist)
