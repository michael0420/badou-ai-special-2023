import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy import shape
import random

'''
椒盐噪声
'''


def add_salt_pepper_noise(src, percetage):
	noise_img = np.copy(src)
	noise_num = int(percetage * src.shape[0] * src.shape[1])  # 添加噪声的点的数量
	for i in range(noise_num):
		randX = random.randint(0, src.shape[0] - 1)
		randY = random.randint(0, src.shape[1] - 1)
		if random.random() <= 0.5:
			noise_img[randX, randY] = 0
		else:
			noise_img[randX, randY] = 255
	return noise_img


img = cv2.imread('lenna.png', 0)
img_noise = add_salt_pepper_noise(img, 0.2)

fig = plt.figure(figsize=(7, 7))
plt.subplot(121), plt.title("Original image "), plt.axis('off')
plt.imshow(img, cmap='gray', vmin=0, vmax=255)  # 原始图像
plt.subplot(122), plt.title("lenna_GaussianNoise "), plt.axis('off')
plt.imshow(img_noise, cmap='gray', vmin=0, vmax=255)  # 原始图像
plt.show()
