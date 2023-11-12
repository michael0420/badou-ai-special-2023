import cv2
import random


def gaussiannoise(src, means, sigma, percentage):
    NoiseImg = src
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img1 = cv2.imread("lenna.png", 0)  # gray
guss_img = gaussiannoise(img1, 6, 8, 0.8)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow("source", img1)
cv2.imshow("source", img2)
cv2.imshow("gaussian noise", guss_img)
cv2.waitKey()
