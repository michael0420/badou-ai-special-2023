import numpy as np
import cv2
import time

def ahash(img,w,d):
    #均值哈希

    #缩放
    imgg = cv2.resize(img,(w,d),interpolation=cv2.INTER_CUBIC)

    #转为灰度图
    gray = cv2.cvtColor(imgg,cv2.COLOR_BGR2GRAY)

    s = np.mean(gray)
    hash = np.where(gray>s,'1','0')
    hash = hash.reshape(1,64)
    hash =  ''.join(hash[0,:])

    return hash
def dhash(img,w,d):
    imgg = cv2.resize(img, (w, d), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，反之置为0，生成感知哈希序列（string）
    for i in range(d):
        for j in range(d):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

def cmp(ahash,dhash):
    if(len(ahash)!=len(dhash)):
        return -1
    hh = [1 for i in range(64) if ahash[i]!=dhash[i]]
    return len(hh)


def test():
    img = cv2.imread('lenna.png')
    ah = ahash(img,8,8)
    print('均值哈希：',ah)
    dh = dhash(img,9,8)
    print('插值哈希：',dh)
    print(cmp(ah,dh))

    img1 = cv2.imread('lenna.png')
    img2 = cv2.imread('lenna_noise.png')
    start_time = time.time()
    for _ in range(1000):
        hash1 = ahash(img1,8,8)
        hash2 = ahash(img2,8,8)
        cmp(hash1, hash2)

    print(">>> ahash执行%s次耗费的时间为%.4f s." % (1000, time.time() - start_time))

    start_time = time.time()
    for _ in range(1000):
        hash1 = dhash(img1,9,8)
        hash2 = dhash(img2,9,8)
        cmp(hash1, hash2)

    print(">>> dhash执行%s次耗费的时间为%.4f s." % (1000, time.time() - start_time))

if __name__ =='__main__':
    test()