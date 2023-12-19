import cv2
import numpy as np


def avgHash(img):
    img = cv2.resize(img, (8, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg = np.mean(gray.flatten())
    hash_str = ''
    for i in range(8):
        for j in range(8):
            hash_str = hash_str + ('1' if gray[i, j] > avg else '0')
    return hash_str


def diffHash(img):
    img = cv2.resize(img, (9, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            hash_str = hash_str + ('1' if gray[i, j] > gray[i, j + 1] else '0')
    return hash_str


def HMDistance(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
hash1 = avgHash(img1)
hash2 = avgHash(img2)
print(hash1)
print(hash2)
print('均值哈希汉明距离：', HMDistance(hash1, hash2))

hash1 = diffHash(img1)
hash2 = diffHash(img2)
print(hash1)
print(hash2)
print('差值哈希汉明距离：', HMDistance(hash1, hash2))
