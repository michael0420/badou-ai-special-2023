import cv2
import numpy as np


def normal_interpolation(img, param):
    # 获取原图片的宽和高
    old_h,old_w ,channels = img.shape
    new_h,new_w = param

    #设置一个要生成的image
    new_image = np.zeros([new_h,new_w,channels],img.dtype)
    # 判断高和宽分别扩大的倍数
    h_mult = old_h/new_h
    w_mult = old_w/new_w
    # 循环要生产图片的宽和高，的每个像素点，然后利用最临近法确定在原 img中的位置
    # 循环3个通道
    for m in range(3):
        # 遍历新图片的纵方向的像素点
        for i in range(new_h):
            #为新图片当前行寻找h 和w的值
            for j in range(new_w):
                # 四舍五入，导致的问题会超过边界
                oldh = np.round(i * h_mult)
                oldw = np.round(j * w_mult)
                # 防止数组越界
                oldh = np.where(oldh >= old_h, (oldh-1), oldh)
                oldw = np.where(oldw >= old_w,  (oldw-1), oldw)
                new_image[i, j, m] = img[oldh.astype(int),oldw.astype(int),m]
    return new_image


if __name__ == '__main__':
    img = cv2.imread('/Users/xiaoluoluosuper-/Downloads/lenna.png')
    dst = normal_interpolation(img, (2400, 2400))
    cv2.imshow('normal interp', dst)
    cv2.imshow('origin image', img)
    cv2.waitKey()