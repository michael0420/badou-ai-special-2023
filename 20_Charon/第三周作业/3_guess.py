import cv2
import numpy as np
# 读取原图获取基础信息
src_img = cv2.imread(r'C:\Users\Administrator\Desktop\homework\lenna.png',0)
h, w = src_img.shape[:2]
j = 0
f = 20
noise = np.random.normal(j, f, [h, w])         # 给正态随机数

dst_img = src_img + noise
dst_img = np.clip(dst_img, 0, 255)
dst_img = dst_img / 255
cv2.imshow('src_img', src_img)
cv2.imshow('dst_img', dst_img)
cv2.waitKey(0)