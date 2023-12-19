import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

'''
高斯噪声
'''


def add_gaussian_noise(src, means, sigma, percetage):  # src（原始图像）、means（高斯噪声均值）、sigma（高斯噪声标准差）和 percetage（噪声比例）
    noise_img = np.copy(src)
    noise_num = int(percetage * src.shape[0] * src.shape[1])  # 添加噪声的点的数量
    for i in range(noise_num):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        noise = random.gauss(means, sigma)
        # 在原有像素灰度值上加上高斯分布的随机数
        pixel_value = noise_img[randX, randY] + noise
        pixel_value = max(0, min(pixel_value, 255))
        noise_img[randX, randY] = pixel_value
    return noise_img


img = cv2.imread('lenna.png', 0)
img_noise = add_gaussian_noise(img, 2, 4, 0.8)


fig = plt.figure(figsize=(7, 7))
plt.subplot(121), plt.title("Original image "), plt.axis('off')
plt.imshow(img, cmap='gray', vmin=0, vmax=255)  # 原始图像
plt.subplot(122), plt.title("lenna_GaussianNoise "), plt.axis('off')
plt.imshow(img_noise, cmap='gray', vmin=0, vmax=255)  # 原始图像
plt.show()
