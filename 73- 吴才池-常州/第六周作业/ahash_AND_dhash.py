import cv2
import numpy as np
def ahash(img): #均值hush
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sum=0 #像素和
    hash_str=''
    for i in range(8):
        for j in range(8):
            sum+=gray[i,j]

    avg=sum/64

    for i in range(8):
        for j in range(8):
            if gray[i,j]>avg:
                hash_str+='1'
            else:
                hash_str+='0'

    return hash_str

def dhash(img):
    #差值哈希
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
    for i in range(8):
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                hash_str+='1'
            else:
                hash_str+='0'
    return hash_str

def cmp(hash1,hash2):
    n=0
    if (len(hash1)!=len(hash2)):
       return -1
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n+=1
    return n

img1=cv2.imread("lenna.png")
img2=cv2.imread("lenna_noise.png")
hash1=ahash(img1)
hash2=ahash(img2)
print(hash2)
print(hash1)
print("相似度",cmp(hash1,hash2))


hash1=dhash(img1)
hash2=dhash(img2)

print(hash1)
print(hash2)

print("相似度",cmp(hash1, hash2))
















