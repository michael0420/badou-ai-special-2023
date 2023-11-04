import numpy as np
import cv2
"""
@author: zhangxi

最近插值原理演示
"""


def function(img):
    h, w, c = img.shape
    print("h:{},w:{},c{}".format(h, w, c))
    empty_img = np.zeros((800, 800, c), np.uint8)
    scalH = 800 / h
    scalW = 800 / w

    for ch in range(c):
        for i in range(800):
            for j in range(800):
                x = int(i / scalH + 0.5)
                y = int(j / scalW + 0.5)
                empty_img[i, j, ch] = img[x, y, ch]

    return empty_img


image = cv2.imread("lenna.png")
img_scal = function(image)
cv2.imshow("scalImg", img_scal)
cv2.waitKey(0)
