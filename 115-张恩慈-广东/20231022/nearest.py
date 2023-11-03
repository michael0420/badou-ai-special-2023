# 作业2：实现最近邻插值法
import cv2
import numpy as np

img = cv2.imread("lenna.png")
cv2.imshow("origin", img)

print("---nearest image lenna----")
h, w, channels = img.shape
# 1、创建一个同样通道数的新图
size = 300
img_dst = np.zeros((size, size, channels), dtype=np.uint8)
# 2、计算横纵缩放比例
height_scale = h / size
width_scale = w / size
for i in range(size):
    for j in range(size):
        # 3、找 新图(i,j)点 对应回原图 (i, j)点 的最近邻点
        i_nearest = int( i * width_scale + 0.5) # +0.5 是四舍五入
        j_nearest = int( j * height_scale + 0.5)
        # 4、将原图的 (i,j) 点 值 赋给新图的 (i, j)
        img_dst[i, j] = img[i_nearest, j_nearest]
cv2.imshow("after nearest", img_dst)
cv2.waitKey(0)  # 等待键盘输入， cv2.imshow需要这一行，不知道为什么

