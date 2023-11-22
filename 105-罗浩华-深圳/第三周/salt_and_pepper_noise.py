import random

import cv2
import numpy as np


def salt_and_pepper_noise(src_img, percentage, ):
    noise_img = src_img.copy()
    height, width = noise_img.shape[:2]
    noise_pixel_num = int(height * width * percentage)
    for i in range(noise_pixel_num):
        # get random pixel row and column
        random_row = random.randint(0, noise_img.shape[0] - 1)
        random_column = random.randint(0, noise_img.shape[1] - 1)
        noise_img[random_row, random_column]=get_noise_pixel_value()
    return noise_img


def get_noise_pixel_value():
    return 255 if random.random() > 0.5 else 0


img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
img_gaussian_noise = salt_and_pepper_noise(img, 0.2)
cv2.imshow('img_salt_and_pepper_noise', np.hstack([img, img_gaussian_noise]))
cv2.waitKey(0)
