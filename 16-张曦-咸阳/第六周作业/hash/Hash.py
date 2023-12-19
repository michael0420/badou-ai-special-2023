import cv2


def aHash(img, width=8, high=8):
    """
    均值hash
    """
    resize_img = cv2.resize(img, (8, 8))
    gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

    sum = 0
    for i in range(width):
        for j in range(high):
            sum = gray[i, j] + sum

    avg = sum / 64
    hash_str = ''
    for i in range(width):
        for j in range(high):
            if gray[i, j] > avg:
                hash_str = hash_str + "1"
            else:
                hash_str = hash_str + "0"

    return hash_str


def dHash(img, width=8, high=8):
    """
    差值hash
    """
    resize_img = cv2.resize(img, (9, 8))
    gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(width):
        for j in range(high):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + "1"
            else:
                hash_str = hash_str + "0"

    return hash_str


def compare(hash1, hash2):
    """
    比较两个hash的相似度
    """
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1

    diff = 0
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            diff = diff + 1

    return diff


img = cv2.imread("lenna.png")
img2 = cv2.imread("lenna_noise.png")
hash1 = aHash(img, 8, 8)
hash2 = aHash(img2, 8, 8)
print("均值hash汉明距离=", compare(hash1, hash2))

hash1 = dHash(img, 8, 8)
hash2 = dHash(img2, 8, 8)
print("差值hash汉明距离=", compare(hash1, hash2))
