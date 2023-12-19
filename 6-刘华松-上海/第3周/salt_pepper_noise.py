#实现椒盐噪声*************************


import numpy as np
import cv2

# 读取图像
image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# 给图像添加椒盐噪声
noise_prob = 0.05
s_vs_p = 0.5
salt_vs_pepper = 0.5
out = np.copy(image)

# 添加椒噪声
num_salt = np.ceil(noise_prob * image.size * s_vs_p)
coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
out[coords] = 255

# 添加盐噪声
num_pepper = np.ceil(noise_prob * image.size * (1. - s_vs_p))
coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
out[coords] = 0

# 显示原始图像和添加椒盐噪声后的图像

cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', out)