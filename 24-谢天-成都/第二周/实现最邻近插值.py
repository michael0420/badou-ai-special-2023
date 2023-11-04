import cv2
import numpy as np


def function(img,out_dim):
    src_h,src_w,channels=img.shape
    dst_h,dst_w=out_dim
    img_init=np.zeros((dst_h,dst_w,channels),img.dtype)
    sh=dst_h/src_h
    sw=dst_w/src_w
    for i in range(dst_h):
        for j in range(dst_w):
            x=int(i/sh+0.5)  #向下取整。
            y=int(j/sw+0.5)
            img_init[i,j]=img[x,y]
    return img_init


img=cv2.imread("C:\\Users\\32496\\Desktop\\lenna.png")
img_near=function(img,(700,700))
print(img_near)
print(img_near.shape)
cv2.imshow("nearest interp",img_near)
cv2.imshow("image",img)
cv2.waitKey(0)
