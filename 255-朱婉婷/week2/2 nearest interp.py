# -*- coding;utf-8 -*-
"""
实现最邻近插值
"""

import cv2
import numpy as np

def nearest_interp(src):
    height,weight,channels = src.shape
    emptyImage = np.zeros((800,800,channels),np.uint8)
    #比例缩放
    sh = height/800
    sw = weight/800
    for i in range(800):
        for j in range(800):
            x = int(i*sh+0.5)
            y = int(j*sw+0.5)
            emptyImage[i,j] = src[x,y]
    return emptyImage

img = cv2.imread('lenna.png')
zoom = nearest_interp(img)
cv2.imshow('img',img)
cv2.imshow('nearest_interp',zoom)
cv2.waitKey(0)
