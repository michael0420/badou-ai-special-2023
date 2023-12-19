
""""
使用接口实现Canny算法
"""
import cv2
import numpy as np

img = cv2.imread('lenna.png',0)
cv2.imshow('lenna',cv2.Canny(img,100,300))
cv2.waitKey()
cv2.destroyAllWindows()