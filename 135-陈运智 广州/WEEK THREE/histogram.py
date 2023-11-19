# 获取灰度直方图
# 获取灰度图像
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png")
h,w =img.shape[:2]
#利用公式获取灰度直方图
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int (m[0]*0.11 +m[1]*0.59 +m[2]*0.3)
# 利用cv2中的方法直接修改,
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 在opencv中图像第一层是G,第二层是R,第三层是B
plt.figure()
plt.hist(img_gray.ravel(),256)
plt.show()
# 彩色图像直方图
# img2 = cv2.imread("lenna.png")
# 图像分层分别计算每一种颜色的直方图
# b,g,r = cv2.split(img2)
# hist_B = cv2.calcHist([b],[0], None, [256], [0, 256])
# hist_G = cv2.calcHist([g],[0], None, [256], [0, 256])
# hist_R = cv2.calcHist([r],[0], None, [256], [0, 256])
# plt.figure(figsize=(10, 6))
# plt.plot(hist_B, color='b')
# plt.plot(hist_G, color='g')
# plt.plot(hist_R, color='r')

img3 = cv2.imread('lenna.png')
chans = cv2.split(img3)
colors = ("b","g","r")
for(chan,colors) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color=colors)
plt.title('Histogram for Each Channel')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

plt.xlim(0,256)