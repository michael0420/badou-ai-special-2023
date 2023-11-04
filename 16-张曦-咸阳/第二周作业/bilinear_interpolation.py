import numpy as np
import cv2

"""
@author: zhangxi

双线性插值原理演示
"""


def function(img, new_img_width, new_img_high):
    h, w, c = img.shape
    result_img = np.zeros((new_img_high, new_img_width, c), img.dtype)
    scalY = h / new_img_high
    scalX = w / new_img_width

    for channel in range(c):
        for out_y in range(new_img_high):
            for out_x in range(new_img_width):
                x = (out_x + 0.5) * scalX - 0.5
                y = (out_y + 0.5) * scalY - 0.5

                x = np.clip(x, 0, w - 1)  # 小于0，等于0；大于宽度w-1,等于w-1
                y = np.clip(y, 0, h - 1)  # 小于0，等于0；大于宽度h-1,等于h-1

                x0 = int(x)
                y0 = int(y)
                x1 = min(x0 + 1, w - 1)  # 确保不要越界
                y1 = min(y0 + 1, h - 1)  # 确保不要越界

                print("({},{})".format(x1, y1))

                Q11 = img[x0, y0, channel]
                Q12 = img[x1, y0, channel]
                Q21 = img[x0, y1, channel]
                Q22 = img[x1, y1, channel]

                A = (x - x0) * Q22 + (x1 - x) * Q21
                B = (x - x0) * Q12 + (x1 - x) * Q11
                P = (y1 - y) * B + (y - y0) * A

                result_img[out_x, out_y, channel] = P

    return result_img


image = cv2.imread("lenna.png")
img_temp = function(image, 800, 800)
cv2.imshow("beauty", img_temp)
cv2.waitKey(0)
