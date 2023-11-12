import cv2
import numpy as np
import random

# 高斯噪声
def GaussianNoise(src, means, sigma, percentage):
    NoiseImg = np.copy(src)
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

img = cv2.imread('lenna.png', 0)  # 读取灰度图像
img1 = GaussianNoise(img, 1, 20, 0.8)

cv2.imshow('GaussianNoise', np.hstack((img, img1)))
cv2.waitKey(0)
cv2.destroyAllWindows()
