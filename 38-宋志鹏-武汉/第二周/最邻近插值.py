import cv2
import numpy as np
def function(img):    #定义函数
    height,width,channels = img.shape    #行，列，通道数
    emptyImage=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh + 0.5)  # 转为整型，向下取整
            y=int(j/sh + 0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage


img=cv2.imread("lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
# # cv2.waitKey(0)

# cv2.resize函数
img=cv2.imread("lenna.png")
shape = img.shape
print(shape)
small=cv2.resize(img,(200,200,))
print(small)
print(small.shape)
cv2.imshow("small",small)
cv2.waitKey(0)
