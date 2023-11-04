import cv2
import numpy as np

src_img = cv2.imread('lenna.png')
h, w, channels = src_img.shape
# 目标图的大小
H = 700
W = 700
# 创建空白图
dst_img = np.zeros([H, W, channels], dtype=np.uint8)
# 变化比例
h_rate = float(h) / H
w_rate = float(w) / W
for c in range(channels):
    for dst_x in range(W):
        for dst_y in range(H):
            # 重合中心
            src_x = int(dst_x + 0.5) * w_rate - 0.5
            src_y = int(dst_y + 0.5) * h_rate - 0.5

            src_x0 = int(src_x)
            src_x1 = min(int(src_x + 1), (w - 1))
            src_y0 = int(src_y)
            src_y1 = min(int(src_y + 1), (h - 1))
            # 用src的已知点算出同直线点
            r1 = (src_x1 - src_x) * src_img[src_x0, src_y0, c] + (src_x - src_x0) * src_img[src_x1, src_y0, c]
            r2 = (src_x1 - src_x) * src_img[src_x0, src_y1, c] + (src_x - src_x0) * src_img[src_x1, src_y1, c]

            dst_img[dst_x, dst_y, c] = (src_y1 - src_y) * r1 + (src_y - src_y0) * r2

cv2.imshow('shuangxian', dst_img)
cv2.waitKey(0)