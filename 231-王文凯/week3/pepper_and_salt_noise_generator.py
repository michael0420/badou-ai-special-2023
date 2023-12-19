import numpy as np
import cv2
import random


def pepper_and_salt_noise_generator(img, percentage):
    h, w = img.shape[:2]
    noise_img, noise_num = np.copy(img), int(percentage * h * w)
    for i in range(noise_num):
        rand_x, rand_y = random.randint(0, h - 1), random.randint(0, w - 1)
        noise_img[rand_x, rand_y] = 0 if random.random() <= 0.5 else 255
    return noise_img


if __name__ == "__main__":
    lenna_gray = cv2.cvtColor(cv2.imread('../Images/lenna.png'), cv2.COLOR_BGR2GRAY)
    lenna_noise = pepper_and_salt_noise_generator(lenna_gray, 0.1)
    cv2.imshow('img_gray', lenna_gray)
    cv2.imshow('noise_img', lenna_noise)
    cv2.waitKey(0)
