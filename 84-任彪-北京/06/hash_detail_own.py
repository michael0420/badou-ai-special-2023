import cv2
import numpy as np

def cmp_hash(hash1, hash2):
    """
    Hash值对比
    :param hash1: 感知哈希序列1
    :param hash2: 感知哈希序列2
    :return: 返回相似度
    """
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1

    return 1 - n / len(hash2)

def dhash(img):
    high,weight = img.shape[:2]
    hash_str = ''
    for i in range(weight-1):
        for j in range(weight-1):
            if img[i, j] > img[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def diffHash(img1, img2):
    #差值中，需要后一个跟前一个比，多以weight多
    high = 8
    width = 9
    img_1 = cv2.resize(img1, (width,high), interpolation=cv2.INTER_CUBIC)
    img_2 = cv2.resize(img2, (width,high), interpolation=cv2.INTER_CUBIC)

    img1_hash = dhash(cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY))
    img2_hash = dhash(cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY))
    diff_sim = cmp_hash(img1_hash,img2_hash)
    print("差值hash相似度",diff_sim)


def ahash(grayImg):
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + grayImg[i, j]

    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if grayImg[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def avgHash(img1, img2):
    high = 8
    width = 8
    img_1 = cv2.resize(img1, (width, high), interpolation=cv2.INTER_CUBIC)
    img_2 = cv2.resize(img2, (width, high), interpolation=cv2.INTER_CUBIC)
    img1_hash = ahash(cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY))
    img2_hash = ahash(cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY))
    avg_sim = cmp_hash(img1_hash, img2_hash)
    print("均值hash相似度", avg_sim)


def phash(img):
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img  # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(32, 32)

    # 把二维list变成一维list
    img_list = vis1.flatten()

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i > avg else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])


def perceptionHash(img1, img2):
    high = 8
    width = 8
    img_1 = cv2.resize(img1, (width, high), interpolation=cv2.INTER_CUBIC)
    img_2 = cv2.resize(img2, (width, high), interpolation=cv2.INTER_CUBIC)
    img1_hash = phash(cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY))
    img2_hash = phash(cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY))
    avg_sim = cmp_hash(img1_hash, img2_hash)
    print("感知hash相似度", avg_sim)
    pass


if __name__ == '__main__':
    img1  = cv2.imread("lenna.png")
    img2 = cv2.imread("lenna_noise.png")
    similarity_diff = diffHash(img1,img2)
    similarity_avg = avgHash(img1,img2)
    similarity_p  = perceptionHash(img1,img2)
    