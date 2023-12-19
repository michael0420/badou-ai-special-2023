import cv2 as cv
import random


def salt_and_pepper(img, percetage):
    h, w = img.shape
    niose_num = int(h * w * percetage)
    for i in range(niose_num):
        # 随机一个点
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        # 随机一个值
        if random.random() <= 0.5:
            img[x, y] = 0
        else:
            img[x, y] = 255

    return img


img = cv.imread('lenna.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img2 = salt_and_pepper(gray, 0.2)

cv.imwrite('salt_and_pepper.png', img2)

# opencv 椒盐噪声
