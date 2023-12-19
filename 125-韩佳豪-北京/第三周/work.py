import cv2 as cv

import numpy as np


#直方图均衡化
def function_1():
    img = cv.imread('lenna.jpg', 0)
    equ = cv.equalizeHist(img)
    res = np.hstack((img, equ))
    cv.imwrite('res.png', res)


# 高斯噪声
def function_2():
    img = cv.imread('lenna.jpg', 0)
    noise = np.random.normal(0, 20, size=img.size).reshape(img.shape[0], img.shape[1], img.shape[2])
    img = img + noise
    np.clip(img, 0, 255)
    img = img / 255
    cv.imshow('gauss noise', img)
    cv.waitKey(0)

# 椒盐噪声
def function_3():
    # 信噪比
    SNR=0.1
    img = cv.imread('lenna.jpg', 0)
    w, h, c = img.shape
    mask = np.random.choice((0,1,2), size=(wh,h,1),p=[SNR, (1-SNR) /2., (1-SNR) / 2.])

    mask = np.repeat(mask, c, axis=2)
    img[mask==1] = 255
    img[mask==2] = 0
    cv.imshow('salt noise', img)
    cv.waitKey(0)