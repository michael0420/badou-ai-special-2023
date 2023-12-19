import cv2

import numpy as np


def interpolate_image(image, param):
    # 获取原本图片的宽和高
    old_h, old_w, channels = image.shape
    new_h, new_w = param

    # 新建1个图片，用于填充
    new_image = np.zeros([new_h, new_w, channels], image.dtype)

    # 判断图片的宽和高变化的倍数
    h_change = old_h / new_h
    w_change = old_w / new_w

    # 循环新产生的图片的三个通道上的长、宽像素点
    for m in range(3):
        for n in range(new_h):
            for k in range(new_w):
                oldh = np.round(n * h_change)
                oldw = np.round(k * w_change)
                oldh = np.where(oldh >= old_h, (oldh - 1), oldh)
                oldw = np.where(oldw >= old_w, (oldw - 1), oldw)
                new_image[n, k, m] = image[int(oldh), int(oldw), m]
    return new_image


if __name__ == "__main__":
    image = cv2.imread("C:/Users/15082/Desktop/lenna.png")
    dst = interpolate_image(image, (2400, 1000))
    cv2.imshow("image", dst)
    print('打印完成')
    cv2.waitKey(0)
