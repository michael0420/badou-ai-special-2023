import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
img = cv2.imread('lenna.png')

# cv2.imread()读取图像时，默认彩色图像的三通道顺序为B、G、R,  plt.show 颜色会很奇怪
# plt.imshow()函数却默认显示图像的通道顺序为R、G、B
# plt.imread 可以解决这个问题 但也可以重新调换顺序如下所示
b,g,r = cv2.split(img) #提取b,g,r
img = cv2.merge([r,g,b])
high,wide,channel = img.shape 


plt.subplot(221) 

plt.imshow(img)

#手动调节灰度图
img_gray = np.zeros([high,wide],img.dtype)
for i in range(high):
    for j in range(wide):
        RGB = img[i,j]
        img_gray[i,j] = int(RGB[0]*0.11 + RGB[1]*0.59 + RGB[2]*0.3)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')



# 二值图
img_gray = rgb2gray(img) #skimage的的调用灰度化（最简单的方法）

img_binary = np.where(img_gray >= 0.5 ,1 ,0)

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')




#plt.xticks([]), plt.yticks([]) # 隐藏x和y轴
plt.show()
