import random

import cv2


def GaussNoise(src, mean, sigma, percetage):
    src_noise_img = src
    src_noise_num = int(src.shape[0] * src.shape[1] * percetage)

    for i in range(src_noise_num):
        random.seed()
        randX = random.randint(0, src.shape[0] - 1)
        random.seed()
        randY = random.randint(0, src.shape[1] - 1)
        src_noise_img[randX, randY] = src[randX, randY] + random.gauss(mean, sigma)
        if src_noise_img[randX, randY] < 0:
            src_noise_img[randX, randY] = 0
        elif src_noise_img[randX, randY] > 255:
            src_noise_img[randX, randY] = 255

    return src_noise_img


img = cv2.imread('lenna.png', 0)
cv2.imshow("src", img)
img_gauss = GaussNoise(img, 2, 20, 0.8)  # mean=1,sigma=0,80%的像素点添加噪声

cv2.imshow("imgnoise", img_gauss)


img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source',img2)

cv2.waitKey(0)
