import cv2
import numpy as np
# 读取原图获取基础信息
src_img = cv2.imread(r'C:\Users\Administrator\Desktop\homework\lenna.png',0)
h, w = src_img.shape[:2]
noise_num = 0.2
dst_img = np.zeros([h, w], src_img.dtype)
# 随机赋值
for i in range(h):
    for j in range(w):
        rad_num = np.random.random()         # 给随机数
        if rad_num < noise_num: # 椒
            dst_img[i, j] = 0
        elif rad_num == (1-noise_num): # 盐
            dst_img[i, j] = 255
        else:
            dst_img[i, j] = src_img[i, j]


cv2.imshow('src_img', src_img)
cv2.imshow('dst_img', dst_img)
cv2.waitKey(0)


dst_img_2 = src_img.copy()
