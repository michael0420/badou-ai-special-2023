import cv2
import numpy as np


def nearest_interpolation(new_height, new_width):
    img = cv2.imread('lenna.png')
    height, width, channel = img.shape
    new_img = np.zeros([new_height, new_width, channel], img.dtype)
    sh = height / new_height
    sw = width / new_width
    for i in range(new_height):
        for j in range(new_width):
            # or img[round(i*sh), round(j*sw)]
            # or img[int(i*sh+0.5), int(j*sw+0.5)]
            new_img[i, j] = img[int(i * sh), int(j * sw)]
    return new_img


if __name__ == '__main__':
    new_img = nearest_interpolation(2000, 2000)
    print(new_img.shape)
    print(new_img)
    cv2.imshow('new_img', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
