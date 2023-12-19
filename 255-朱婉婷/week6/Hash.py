"""
实现均值哈希和差值哈希
"""

import cv2
import numpy as np


#均值哈希
def aHash(img, width=8, high=8):
    #cv2.INTER_CUBIC双三次差值，改变形状
    img = cv2.resize(img,(width,high),interpolation=cv2.INTER_CUBIC)

    #转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #求平均灰度
    avg = np.mean(gray)
    
    #均值哈希
    hash_str =  ''
    img_list = gray.flatten()
    for i in img_list:
        if i > avg:
            hash_str += '1'
        else:
            hash_str += '0'

    return hash_str

#差值哈希
def dHash(img, width=9, high=8):
    img = cv2.resize(img, (width,high),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #差值哈希
    hash_str = ''
    for i in range(high):
        for j in range(width-1):
            if gray[i,j] > gray[i,j+1]:
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str

#哈希值对比
def cmpHash(hash1,hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    
    return 1 - n/len(hash2)


img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')

hash1 = aHash(img1)
hash2 = aHash(img2)
print("均值哈希相似度：",cmpHash(hash1,hash2))

hash1 = dHash(img1)
hash2 = dHash(img2)
print("差值哈希相似度：",cmpHash(hash1,hash2))
