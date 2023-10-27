import cv2
import numpy as np

# 最临近插值
# 读取图片基础信息
src_img = cv2.imread('lenna.png')
h, w, channels = src_img.shape
# 创建目标空白图
H, W = 800, 800
dst_img = np.zeros([H, W, channels], np.uint8)
# 计算转换比例
h_rate = float(h) / H
w_rate = float(w) / W
# 为空白图手动赋值
for c in range(3):
    for i in range(H):
        for j in range(W):
            x = int(i * h_rate + 0.5)
            y = int(j * w_rate + 0.5)
            dst_img[i, j, c] = src_img[x, y, c]
# 输出图片
cv2.imshow('nearleast', dst_img)
cv2.waitKey(0)
