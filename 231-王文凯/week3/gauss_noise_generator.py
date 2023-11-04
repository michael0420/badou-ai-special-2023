import cv2
import numpy as np
import random


# img 图片 mean 均值 sigma 标准差 percentage 图片受噪声影响范围 百分比
def gauss_noise_generator(img, mean, sigma, percentage):
    h, w = img.shape[:2]
    noise_img, noise_num = np.copy(img), int(percentage * h * w)
    for i in range(noise_num):
        rand_x, rand_y = random.randint(0, h - 1), random.randint(0, w - 1)
        noise_img[rand_x, rand_y] = noise_img[rand_x, rand_y] + random.gauss(mean, sigma)
        if noise_img[rand_x, rand_y] < 0:
            noise_img[rand_x, rand_y] = 0
        elif noise_img[rand_x, rand_y] > 255:
            noise_img[rand_x, rand_y] = 255
    return noise_img


if __name__ == "__main__":
    lenna = cv2.imread('../Images/lenna.png')
    lenna_gray = cv2.cvtColor(lenna, cv2.COLOR_BGR2GRAY)
    gauss_img = gauss_noise_generator(lenna_gray, 2, 4, 0.9)
    cv2.imshow('img_gray', lenna_gray)
    cv2.imshow('gauss_img', gauss_img)
    cv2.waitKey(0)
