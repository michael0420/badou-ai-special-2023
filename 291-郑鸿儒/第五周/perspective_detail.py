#!/use/bin/env python
# encoding=utf-8
import cv2
import numpy as np
from warpMatrix import WarpPerspectiveMatrix

img = cv2.imread("./img/photo1.jpg")

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])


warpMatrix = WarpPerspectiveMatrix(src, dst)
new_img = np.zeros((488, 337, 3), dtype=np.uint8)
for y in range(new_img.shape[0]):
    for x in range(new_img.shape[1]):
        matrix_cur = [x, y, 1]
        res = np.dot(np.array(warpMatrix.I), matrix_cur).ravel()
        src_x = int(res[0] / res[2])
        src_y = int(res[1] / res[2])
        if 0 <= src_x < 540 and 0 <= src_y < 960:
            # 直接简单粗暴的赋值反而效果好点，猜测是因为做透视变换时本身就有误差，
            # 导致回代时所有点不能一一对应，做双线行插值算四个点的影响反而多此一举
            # 直接四舍五入或截取整数部分反而更接近实际情况
            new_img[y, x] = img[round(res[1] / res[2]), round(res[0] / res[2])]

        # 双线性插值方法
        # 求目标点对应原来点相对于原图中src_x， src_y的偏移量
        #     dx = res[0] / res[2] - src_x
        #     dy = res[1] / res[2] - src_y
        #     src_x1 = min(src_x + 1, new_img.shape[1] - 1)
        #     src_y1 = min(src_y + 1, new_img.shape[0] - 1)
        #     for k in range(3):
        #         # 双线性插值计算目标点对应点的rgb值
        #         tmp1 = dx * img[src_y, src_x1, k] + (1 - dx) * img[src_y, src_x, k]
        #         tmp2 = dx * img[src_y1, src_x1, k] + (1 - dx) * img[src_y1, src_x, k]
        #         tmp = dy * tmp2 + (1 - dy) * tmp1
        #         new_img[y, x, k] = int(tmp)
cv2.imshow('new image', new_img)
cv2.waitKey(0)
