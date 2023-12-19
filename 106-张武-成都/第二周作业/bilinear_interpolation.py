import cv2 as cv
import numpy as np


def bilinear(img, size):
    # 原图宽高通道
    src_h, src_w, channels = img.shape
    # 目标图宽高
    dst_h, dst_w = size[1], size[0]

    # 检查图片大小是否一致
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    # 创建一个目标图容器
    dst_img = np.zeros((dst_h, dst_w, channels), dtype=np.uint8)

    # 计算比例
    scale_h = src_h / dst_h
    scale_w = src_w / dst_w

    for i in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 通过目标图的点dst_x,dst_y计算在原图上该点所在位置
                # 原图所在位置x = 目标图x * (原图宽度/目标图宽度)
                # 原图所在位置y = 目标图y * (原图高度/目标图高度)
                # 几何中心对称
                # 原图所在位置x = (目标图x + 0.5) * (原图宽度/目标图宽度) - 0.5
                # 原图所在位置y = (目标图y + 0.5) * (原图高度/目标图高度) - 0.5
                src_x = (dst_x + 0.5) * scale_w - 0.5
                src_y = (dst_y + 0.5) * scale_h - 0.5

                # 获取最邻近的4个点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                # 计算要插入的像素值

                # (x0,y1)   (x,y1),r2   (x1,y1)
                #
                # (x0,y)    (x,y)       (x1, y)
                #
                # (x0,y0)   (x,y0),r1   (x1,y0)

                # 在x方向做插值计算
                r1 = img[src_y0, src_x1, i] * (src_x - src_x0) + img[src_y0, src_x0, i] * (src_x1 - src_x)
                r2 = img[src_y1, src_x0, i] * (src_x1 - src_x) + img[src_y1, src_x1, i] * (src_x - src_x0)

                # 在y方向做插值计算
                y_val = (src_y1 - src_y) * r1 + (src_y - src_y0) * r2

                # 写入目标图
                dst_img[dst_y, dst_x, i] = int(y_val)

    return dst_img


img = cv.imread('lenna.png')

dst_img = bilinear(img, (800, 800))

cv.imshow('双线性插值图', dst_img)
cv.waitKey(0)

cv.imwrite('bilinear_interpolation.png', dst_img)

# opencv双线性插值
dst_img = cv.resize(img, (800, 800), interpolation=cv.INTER_LINEAR)
cv.imwrite('bilinear_interpolation_cv.png', dst_img)
