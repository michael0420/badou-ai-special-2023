import cv2
import numpy

# cv2读取的图片，bgr


def get_gray(img):
    h, w = img.shape[:2]
    img_gray = numpy.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            img_gray[i, j] = int(0.3 * r + 0.59 * g + 0.11 * b)
    return img_gray


def get_binaryzation(img_gray):
    h, w = img_gray.shape[:2]
    img_binaryzation = numpy.zeros([h, w], img_gray.dtype)
    for i in range(h):
        for j in range(w):
            img_binaryzation[i][j] = 0 if img_gray[i, j] <= 0.5 else 1
    return img_binaryzation


if __name__ == "__main__":
    lenna = cv2.imread('../Images/lenna.png')
    lenna_gray = cv2.cvtColor(lenna, cv2.COLOR_BGR2GRAY)
    lenna_gray_1 = get_gray(lenna)
    # 进行二值化前先对图像进行归一化处理
    lenna_gray_max, lenna_gray_min = numpy.max(lenna_gray_1), numpy.min(lenna_gray_1)
    lenna_gray_normalize = (lenna_gray - lenna_gray_min) / (lenna_gray_max - lenna_gray_min)

    lenna_binaryzation = get_binaryzation(lenna_gray_normalize)

    print('lenna_gray', lenna_gray)
    print('lenna_gray_1', lenna_gray_1)
    print('lenna_binaryzation', lenna_binaryzation)

    # cv2.imshow('lenna_gray', lenna_gray)
    # cv2.imshow('lenna_gray_1', lenna_gray_1)
    cv2.imshow('lenna_binaryzation', lenna_binaryzation)
    cv2.waitKey()


