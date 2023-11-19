import random

import cv2
import numpy as np


def gaussian_noise(src_img, mean, sigma, percentage, ):
    noise_img = src_img.copy()
    height, width = noise_img.shape[:2]
    noise_pixel_num = int(height * width * percentage)
    for i in range(noise_pixel_num):
        # get random pixel row and column
        random_row = random.randint(0, noise_img.shape[0] - 1)
        random_column = random.randint(0, noise_img.shape[1] - 1)
        noise_pixel_value = get_noise_pixel_value(mean, noise_img, random_row, random_column, sigma)
        while noise_pixel_value < 0 or noise_pixel_value > 255:
            noise_pixel_value = get_noise_pixel_value(mean, noise_img, random_row, random_column, sigma)
        noise_img[random_row, random_column] = noise_pixel_value
    return noise_img


def get_noise_pixel_value(mean, noise_img, random_row, random_column, sigma):
    return noise_img[random_row, random_column] + random.gauss(mean, sigma)


img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
img_gaussian_noise = gaussian_noise(img, 20, 20, 0.9)
cv2.imshow('img_gaussian_noise', np.hstack([img, img_gaussian_noise]))
cv2.waitKey(0)
