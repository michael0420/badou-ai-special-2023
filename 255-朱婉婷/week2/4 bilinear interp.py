# -*- coding:utf-8 -*-

"""
双线性插值
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

#输入参数：输入图像矩阵+输出矩阵大小
def bilinear_interp(img, out_dim):
    src_h, src_w = img.shape[:2]
    dst_h, dst_w = out_dim[0], out_dim[1]
    print('src_h, src_w:',src_h,src_w)
    print('dst_h, dst_w:',dst_h,dst_w)
    if src_h==dst_h and src_w==dst_w:
        return img.copy()
    #输出图像矩阵
    dst_img = np.zeros((dst_h,dst_w,3),np.uint8)
    scale_x, scale_y =  float(src_w)/dst_w, float(src_h)/dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #中心对称
                src_x = (dst_x+0.5)*scale_x-0.5
                src_y = (dst_y+0.5)*scale_y-0.5
                #确定周围一像素的点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1, src_w-1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1,src_h-1)
                #对x方向线性插值
                temp0 = (src_x1-src_x)*img[src_y0,src_x0,i]+(src_x-src_x0)*img[src_y0,src_x1,i]
                temp1 = (src_x1-src_x)*img[src_y1,src_x0,i]+(src_x-src_x0)*img[src_y1,src_x1,i]
                #对y方向线性插值
                dst_img[dst_y,dst_x,i] = int((src_y1-src_y)*temp0+(src_y-src_y0)*temp1)
    return dst_img

img = cv2.imread('lenna.png')
img_bili = bilinear_interp(img,(700,700))
cv2.imshow('img',img)
cv2.imshow('bilinear interp',img_bili)
cv2.waitKey(0)


