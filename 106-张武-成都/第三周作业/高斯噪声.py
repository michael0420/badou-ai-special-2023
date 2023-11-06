import cv2 as cv
import random


def gauss_noise(src, mean, sigma, percetage):
    niose_img = src
    h, w = src.shape
    niose_num = int(h * w * percetage)
    for i in range(niose_num):
        # 随机一个点
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        # 生成高斯随机数
        gauss_num = random.gauss(mean, sigma)
        # 给随机点加上高斯随机数
        niose_img[x, y] = niose_img[x, y] + gauss_num

        if niose_img[x, y] < 0:
            niose_img[x, y] = 0
        elif niose_img[x, y] > 255:
            niose_img[x, y] = 255

    return niose_img


img = cv.imread('lenna.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gauss_img = gauss_noise(gray, 2, 4, 0.8)

cv.imwrite('gauss.png', gauss_img)
