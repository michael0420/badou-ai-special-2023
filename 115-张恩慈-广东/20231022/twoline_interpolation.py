# 作业4：实现双线性插值
import cv2
import numpy as np

img = cv2.imread('lenna.png')
cv2.imshow("origin", img)
h, w, channels = img.shape
# 1、创建一个单通道新图
scale = 1.5
dst_h = h * scale
dst_w = w * scale
img_dst = np.zeros((dst_h, dst_w, channels), img.dtype)
scale_width = float(w)/ dst_w
scale_height = float(h)/ dst_h

for channel in range(channels):
    for x in range(dst_w):
        for y in range(dst_h):
            # 以两图中心重合点作为大前提进行缩放
            # 先找到新图对应原图的点(core_x, core_y)
            core_x = (x + 0.5) * scale_width - 0.5
            core_y = (y + 0.5) * scale_height - 0.5
            # 找到与该点关联的原图4个临近点，为双线性插值做准备
            # (src_x1, src_y1)
            # (src_x1, src_y2)
            # (src_x2, src_y1)
            # (src_x2, src_y2)
            src_x1 = int(core_x - 0.5)
            src_x2 = min(src_x1 + 1, w - 1)

            src_y1 = int(core_y - 0.5)
            src_y2 = min(src_y1 + 1, h -1)
            # 根据双线性插值公式，先在x方向做插值，计算出 f(R2) 和 f(R1)
            R1_y = ((src_x2 - core_x)/(src_x2 - src_x1)) * img[src_x1, src_y1, channel] + ((core_x - src_x1)/(src_x2 - src_x1)) * img[src_x2, src_y1, channel]
            R2_y = ((src_x2 - core_x)/(src_x2 - src_x1)) * img[src_x1, src_y2, channel] + ((core_x - src_x1)/(src_x2 - src_x1)) * img[src_x2, src_y2, channel]
            # 再在y方向做插值，计算出f(P)==新图(x, y)点值
            img_dst[x, y, channel] = ((src_y2 - core_y)/(src_y2 - src_y1)) * R1_y + ((core_y - src_y1)/(src_y2 - src_y1)) * R2_y

cv2.imshow("after twoline interpolation", img_dst)
cv2.waitKey(0)  # 等待键盘输入， cv2.imshow需要这一行，不知道为什么
