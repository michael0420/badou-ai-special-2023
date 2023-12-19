'''

@author Songhua Zou
Dies ist eine der Hausaufgaben von Songhua.

Please do not apply "Ctrl C + V" magic to my code without my permission.
'''

import cv2
import numpy as np
import random

def gaussian(img, means, sigma, pcnt):
    ori = img
    noised = img.copy()
    nos_num = int(pcnt * ori.shape[0] * ori.shape[1])

    for i in range(nos_num):

        rand_x = random.randint(0, ori.shape[0] - 1)
        rand_y = random.randint(0, ori.shape[1] - 1)

        # For a color image, we need to process each channel separately
        for channel in range(ori.shape[2]):
            # Add Gaussian noise at the current position for the current channel
            noise = random.gauss(means, sigma)
            noised_value = noised[rand_x, rand_y, channel] + noise
            # Ensure the value stays within the range 0 to 255
            noised_value = np.clip(noised_value, 0, 255)
            noised[rand_x, rand_y, channel] = noised_value

    cv2.imshow('Compare Before & After Gaussian Noised', np.hstack([ori, noised]))
    cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    gaussian(img, 2, 0.1, 0.1)