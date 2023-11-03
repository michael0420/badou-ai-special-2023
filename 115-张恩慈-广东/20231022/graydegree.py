# 作业1：实现灰度化
# 使用浮点算法 Gray = 0.3R + 0.59G + 0.11B
import cv2
import numpy as np

img = cv2.imread("lenna.png")
cv2.imshow("origin", img)

print("---graydegree image lenna----")
h, w = img.shape[:2]
# 1、创建一个单通道新图
img_dst = np.zeros((h, w, 1), dtype=np.uint8)
# 2、给新图矩阵赋值
for i in range(h):
    for j in range(w):
        three_degree = img[i, j] # BGR三通道
        # 3、给新图矩阵赋每一个点的色值
        img_dst[i, j] = three_degree[0] * 0.11 + three_degree[1] * 0.59 + three_degree[2] * 0.3
cv2.imshow("after graydegree", img_dst)
cv2.waitKey(0)  # 等待键盘输入， cv2.imshow需要这一行，不知道为什么