import cv2
import numpy as np


def function(img):
    height, width, channels = img.shape

    new_Image = np.zeros((800, 800, channels), np.uint8)

    sh = 800 / height
    sw = 800 / width

    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)  # int(),转为整型，使用向下取整。
            y = int(j / sw + 0.5)
            new_Image[i, j] = img[x, y]
    return new_Image

image = cv2.imread("lenna.png")
cv2.imshow("image", image)
nearest_image = function(image)
cv2.imshow("nearest interp", nearest_image)
cv2.waitKey(0)


