import cv2
import numpy as np


# 最邻近插值法

def function(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((300, 300, channels), np.uint8)
    sh = 300 / height
    sw = 300 / width

    for i in range(300):
        for j in range(300):
            x = int(i / sh)
            y = int(j / sw)
            emptyImage[i, j] = img[x, y]
    return emptyImage


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    zoom = function(img)
    print(zoom.shape)
    cv2.imshow("nearest interp", zoom)
    cv2.imshow("image", img)
    cv2.waitKey(0)

