import cv2
import numpy as np
from matplotlib import pyplot as plt


# 灰度图像直方图
# 获得灰度图像
img1 = cv2.imread("lenna.png",1)
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# 灰度图像直方图
plt.figure()
plt.hist(gray.ravel(),256)
plt.show()

# 灰度图像直方图均衡化
dst1 = cv2.equalizeHist(gray)

hist = cv2.calcHist([dst1],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst1.ravel(),256)
plt.show()
cv2.imshow("Histogram Equalization",np.hstack([gray,dst1]))
# cv2.waitKey(0)




# 彩色图像直方图
img2 = cv2.imread("lenna.png",1)
cv2.imshow("src",img2)

chans = cv2.split(img2)
colors = ("b","g","r")
plt.figure()
plt.title("Color Histogram")
plt.xlabel("bins")
plt.ylabel("# of Pixels")

for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])
plt.show()

# 彩色图像直方图均衡化
# 分解通道，对每个通道均衡化
(b,g,r) = cv2.split(img2)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# 合并每个通道
result = cv2.merge((bH,gH,rH))
cv2.imshow("dst2",result)
cv2.waitKey(0)