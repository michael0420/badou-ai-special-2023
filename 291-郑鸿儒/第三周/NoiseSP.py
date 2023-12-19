import cv2
# import random
from numpy import random


def noise_sp(src, percentage=1):
    NoiseImg = src
    NoiseNum = int(NoiseImg.shape[0] * NoiseImg.shape[1] * percentage)

    for i in range(NoiseNum):
        randX = random.randint(0, NoiseImg.shape[1] - 1)
        randY = random.randint(0, NoiseImg.shape[0] - 1)

        NoiseImg[randX, randY] = 0 if random.random() < 0.5 else 255

    return NoiseImg


img = cv2.imread("img/lenna.png", 0)
cv2.imshow("source", img)
noise_img = noise_sp(img, 1)
cv2.imshow("SPNoise", noise_img)
cv2.waitKey()
