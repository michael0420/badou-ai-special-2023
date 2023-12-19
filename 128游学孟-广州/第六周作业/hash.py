import cv2
import numpy as np
import time
import os.path as path

def aHash(img,width = 8,high=8):
    """
    均值哈希算法
    :param img:图像数据
    :param width:图像缩放的宽度
    :param high:图像缩放的高度
    :return:感知哈希序列
    """
    # 缩放为8*8
    img = cv2.resize(img,(width,high),interpolation = cv2.INTER_CUBIC)
    # 转为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s+gray[i,j]

    # 求平均灰度
    avg = s/64
    # 求灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str

def dHash(img,width = 9,high = 8):
    """
    差值感知算法
    :param img:图像数据
    :param width: 图像缩放后的宽度
    :param high: 图像缩放后的高度
    :return: 差值哈希序列
    """

    # 缩放8*8
    img = cv2.resize(img,(width,high),interpolation=cv2.INTER_CUBIC)

    # 转为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，反之置为0，生成感知哈希序列（string）
    for i in range(high):
        for j in range(high):
                if gray[i,j] > gray[i,j+1]:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str+'0'
    return hash_str

def cmp_hash(hash1,hash2):
    """
    Hash值对比
    :param hash1:感知哈希序列1
    :param hash2: 感知哈希序列2
    :return:返回相似度
    """
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        # 不相等则n计数+1
        if hash1[i] != hash2[i]:
            n = n+1
    return 1-n/len(hash2)


def concat_info(type_str,score,time):
    temp = '%s相似度：%.2f %% -----time=%.4f ms' % (type_str,score*100,time)
    print(temp)
    return temp

def test_diff_hash(img1_path,img2_path,loops=1000):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    start_time = time.time()

    for _ in range(loops):
        hash1 = dHash(img1)
        hash2 = dHash(img2)
        cmp_hash(hash1,hash2)
    print(">>> 执行%s次耗费的时间为%.4f s." %(loops,time.time() - start_time))

def test_aHash(img1,img2):
    time1 = time.time()
    hash1 = aHash(img1)
    hash2 = aHash(img2)
    n = cmp_hash(hash1,hash2)
    return concat_info("均值哈希算法",n,time.time() - time1)+"\n"

def test_dHash(img1,img2):
    time1 = time.time()
    hash1 = dHash(img1)
    hash2 = dHash(img2)
    n = cmp_hash(hash1,hash2)
    return concat_info("差值哈希算法",n,time.time() - time1)+"\n"


def deal(img1_path, img2_path):
    info = ''

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 计算图像哈希相似度
    info = info + test_aHash(img1, img2)
    info = info + test_dHash(img1, img2)
    return info

def main():
    img1_path = 'D://xuexi//zuoye//week6//img1.jpg'
    img2_path = 'D://xuexi//zuoye//week6//img2.jpg'
    deal(img1_path,img2_path)

if __name__ == '__main__':
    main()

