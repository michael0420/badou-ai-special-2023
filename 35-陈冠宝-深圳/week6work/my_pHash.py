import cv2
import numpy as np


def pHash(img, width=64, high=64):
    """
    感知哈希算法
    :param img_file: 图像数据
    :param width: 图像缩放后的宽度
    :param high:图像缩放后的高度
    :return:图像感知哈希序列
    """
    # 加载并调整图片为32x32灰度图片
    # img = cv2.imread(img_file, 0)
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建二维列表
    h, w = gray.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = gray  # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(width, high)

    # 把二维list变成一维list
    img_list = vis1.flatten()

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    hash_str = ''
    for i in range(width):
        for j in range(high):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    # avg_list = ['0' if i > avg else '1' for i in img_list]

    # 得到哈希值
    return hash_str
    # return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, width * high, 4)])


# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
hash1 = pHash(img1,32,32)
hash2 = pHash(img2,32,32)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('感知哈希算法相似度：', n)
