import cv2
import numpy as np


def bilinear_interpolation(img, param):
    old_h, old_w, channels = img.shape
    new_h,new_w = param
    new_image = np.zeros((new_h, new_w, channels),img.dtype)
    # 判断高和宽分别扩大的倍数
    h_mult = old_h / new_h
    w_mult = old_w / new_w
    #循环3个通道
    for i in range(channels):
        for h in range(new_h):
            for w in range(new_w):
                # 找到映射到原来图片的点，但是原来图片这个点是不存在的，需要用四周的四个实际存在的点来 逼定这个虚拟点的实际值
                old_x = (w + 0.5) * w_mult - 0.5
                old_y = (h + 0.5) * h_mult - 0.5
                # # 先判断四个点的坐标 ，
                old_y0 = int(old_y)
                # 像素点间隔为1，需要防止数组越界
                old_y1 = np.where(old_y0 >= old_h-1, old_h-1,old_y0+1)
                old_x0 = int(old_x)
                old_x1 = np.where(old_x0 >= old_w-1, old_w-1,old_x0+1)
                # # 带入公式就行
                temp0 = (old_x1 - old_x) * img[old_y0, old_x0, i] + (old_x - old_x0) * img[old_y0, old_x1, i]
                temp1 = (old_x1 - old_x) * img[old_y1, old_x0, i] + (old_x - old_x0) * img[old_y1, old_x1, i]
                new_image[h, w, i] = int((old_y1 - old_y) * temp0 + (old_y - old_y0) * temp1)

    return new_image


if __name__ == '__main__':
    img = cv2.imread('/Users/xiaoluoluosuper-/Downloads/lenna.png')
    new_img = bilinear_interpolation(img, (200, 200))
    cv2.imshow('bilinear img', new_img)
    cv2.imshow('origin img', img)
    cv2.waitKey()