import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread("C:\\Users\\32496\\Desktop\\lenna.png")
#灰度化
img_g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#灰度图直方均衡化
img_g_h=cv2.equalizeHist(img_g)


#分别对三个通道进行直方均衡化然后合并
b,g,r=cv2.split(img)
bh,gh,rh=cv2.equalizeHist(b),cv2.equalizeHist(g),cv2.equalizeHist(r)
img_h=cv2.merge((bh,gh,rh))

#灰度图直方图
hist_g=cv2.calcHist([img_g_h],[0],None,[256],[0,256])
plt.figure()
plt.plot(hist_g)
plt.show()

#彩色直方图
chans=cv2.split(img)
colors=("b","g","r")
plt.figure()
for (chan,color) in zip(chans,colors):
    hist=cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])
plt.show()


cv2.imshow("灰度图直方均衡化",np.hstack([img_g,img_g_h]))
cv2.imshow("彩色图直方均衡化",np.hstack([img,img_h]))
cv2.waitKey(0)

