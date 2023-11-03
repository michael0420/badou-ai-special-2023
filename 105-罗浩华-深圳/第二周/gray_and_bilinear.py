from skimage.color import rgb2gray
import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def show_gray():
    image = cv2.imread('lenna.png')
    height, width = image.shape[:2]
    img_gray_float = np.zeros([height, width], image.dtype)
    img_gray_shift = np.zeros([height, width], image.dtype)
    img_avg = np.zeros([height, width], image.dtype)
    for i in range(height):
        for j in range(width):
            bgr_value = image[i, j]
            img_gray_float[i, j] = int(bgr_value[0] * 0.11 + bgr_value[1] * 0.59 + bgr_value[2] * 0.3)
            img_gray_shift[i, j] = int(bgr_value[0] * 28 + bgr_value[1] * 151 + bgr_value[2] * 76) >> 8
            img_avg[i, j] = int((int(bgr_value[0]) + int(bgr_value[1]) + int(bgr_value[2])) / 3)
    print("Image Gray:")
    cv2.imshow('img_gray_float', img_gray_float)

    cv2.imshow('img_gray_shift', img_gray_shift)

    cv2.imshow('img_avg', img_avg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.subplot(221)
    img = plt.imread("lenna.png")
    # img = cv2.imread("lenna.png", False)
    plt.imshow(img)
    print("---image lenna----")
    print(img)
    img_gray_cv2 = rgb2gray(image)
    plt.subplot(222)
    plt.imshow(img_gray_cv2, cmap='gray')


    print('Image Gray Cv2:')
    print(img_gray_cv2)

    img_binary = np.where(img_gray_cv2 >= 0.5, 1, 0)
    print("-----imge_binary------")
    print(img_binary)
    print(img_binary.shape)
    plt.subplot(223)
    plt.imshow(img_binary, cmap='gray')
    plt.show()


if __name__ == '__main__':
    show_gray()
