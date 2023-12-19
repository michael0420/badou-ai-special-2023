import cv2
import numpy as np


def aHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0

    for i in range(8):
        for j in range(8):
            s += img_gray[i, j]

    avg = s / 64
    hash = ''
    for i in range(8):
        for j in range(8):
           if img_gray[i, j] > avg:
               hash += '1'
           else:
               hash += '0'
    return hash

def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > img_gray[i, j + 1]:
                hash += '1'
            else:
                hash += '0'
    return hash


def compareHash(hash_1, hash_2):
    if len(hash_1) != len(hash_2):
        return -1

    n = 0
    for i in range(len(hash_1)):
        if hash_1[i] != hash_2[i]:
            n += 1
    return n


if __name__ == '__main__':
    img_1 = cv2.imread('iphone1.png')
    img_2 = cv2.imread('iphone2.png')
    a_1 = aHash(img_1)
    a_2 = aHash(img_2)
    print(compareHash(a_1, a_2))

    d_1 = dHash(img_1)
    d_2 = dHash(img_2)
    print(compareHash(d_1, d_2))
