#图像灰度化和二值化

import cv2
from skimage.color import rgb2gray
import numpy as np
import  matplotlib.pyplot as plt
from  PIL import  Image

#灰度化
img=cv2.imread("lenna.png")
h,w=img.shape[:2]
img_gray=np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m=img[i,j]
        img_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)

print(img_gray)
print("image show gray:%s"%img_gray)
cv2.imshow("image show gray",img_gray)

plt.subplot(221)  #两行两列第一个位置
img=plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna---")
print(img)

img_gray=rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
print('---image gray---')
print(img_gray)

#二值化
rows,cols=img_gray.shape
for i in range(rows):
    for j in range(cols):
        if(img_gray[i,j]<=0.6):
            img_gray[i,j]=0 #单通道全0为黑色
        else:
            img_gray[i,j]=1

plt.subplot(223)
plt.imshow(img_gray,cmap="gray")

plt.show()#阻塞作用 显示图像
