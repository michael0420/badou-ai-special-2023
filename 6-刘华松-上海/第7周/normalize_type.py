import cv2
import numpy as np

# 读取图像
img = cv2.imread('lenna.png',0)

# 归一化方式1：使用cv2.normalize函数
normalized_img1 = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 显示归一化后的图像
cv2.imshow('Normalized Image 1', normalized_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()