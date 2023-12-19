import cv2


# 均值哈希算法

def mean_hash(img):
    # 插值法将图片缩放为（8， 8）
    img_88 = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    img_88_gray = cv2.cvtColor(img_88, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    total = 0
    # 求均值
    for i in range(8):
        total += sum(img_88_gray[i])
    mean = total / 8 * 8

    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if img_88_gray[i, j] > mean:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str


# 差值算法

def differ_hash(img):
    # 插值法将图片缩放为（8， 9）,9个两两作差就是8个
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img_89_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''

    # 求差值，每行后一个数大于前一个数为1，相反为0
    for i in range(8):
        for j in range(8):
            if img_89_gray[i, j] > img_89_gray[i, j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str


# Hash值对比

def cmp_hash(hash1, hash2):
    # 长度不一致，无法比较
    if len(hash1) != len(hash2):
        return -1

    n = 0
    # 记录遍历每个字符的比较结果
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n


if __name__ == '__main__':
    img1 = cv2.imread('lenna.png')
    img2 = cv2.imread('lenna_noise.png')
    img1_mean_hash = mean_hash(img1)
    img2_mean_hash = mean_hash(img2)

    img1_differ_hash = differ_hash(img1)
    img2_differ_hash = differ_hash(img2)

    mean_same = cmp_hash(img1_mean_hash, img2_mean_hash)
    print('均值哈希算法相似度：', mean_same)
    differ_same = cmp_hash(img1_differ_hash, img2_differ_hash)
    print('差值哈希算法相似度：', differ_same)


