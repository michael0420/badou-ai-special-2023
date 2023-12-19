'''

@author Songhua Zou
Dies ist eine der Hausaufgaben von Songhua.

Please do not apply "Ctrl C + V" magic to my code without my permission.
'''

import cv2
import numpy as np
import random

def pepper_salt(img, pcnt):
    ori = img
    noised = img.copy()
    nos_num = int(pcnt * ori.shape[0] * ori.shape[1])

    for i in range(nos_num):

        rand_x = random.randint(0, ori.shape[0]-1)
        rand_y = random.randint(0, ori.shape[1] - 1)

        if random.random() <= 0.5:
            noised[rand_x, rand_y] = 0
        else:
            noised[rand_x, rand_y] = 255

    cv2.imshow('Compare Before & After Pepper & Salt Noised', np.hstack([ori, noised]))
    cv2.waitKey(0)


if __name__ == '__main__':

    img = cv2.imread('lenna.png')
    pepper_salt(img, 0.01)



