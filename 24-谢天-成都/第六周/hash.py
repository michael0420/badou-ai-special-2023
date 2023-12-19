import cv2


def aHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s += img_gray[i, j]

    avg = s / 64

    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str



def dHash(img):

    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > img_gray[i, j + 1]:
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str



def cmpHash(hash_1, hash_2):

    n = 0

    if len(hash_1) != len(hash_2):
        return -1

    for i in range(len(hash_1)):
        if hash_1[i] != hash_2[i]:
            n += 1
    return n



img1=cv2.imread('lenna.png')
img2=cv2.imread('lenna_noise.png')
hash1= aHash(img1)
hash2= aHash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print('均值哈希算法相似度：',n)
  
hash1= dHash(img1)
hash2= dHash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print('差值哈希算法相似度：',n)
