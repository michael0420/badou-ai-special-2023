import cv2
import numpy as np

def Mean_hash(image, hash_size=8):
    # 缩放图像
    image = cv2.resize(image, (hash_size, hash_size))

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算像素平均值
    mean = np.mean(gray)

    # 生成二值哈希
    hash_value = (gray > mean).astype(int)

    return hash_value.flatten()

def Difference_Hash(image, hash_size=8):
    # 缩放图像
    image = cv2.resize(image, (hash_size, hash_size))

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算DCT
    dct_result = cv2.dct(np.float32(gray))

    # 取左上角的8x8区块
    dct_result = dct_result[:hash_size, :hash_size]

    # 生成二值哈希
    hash_value = (dct_result > 0).astype(int)

    return hash_value.flatten()

def hamming_distance(hash1, hash2):
    # 计算汉明距离
    return np.sum(hash1 != hash2)

# 读取图像
image1 = cv2.imread("lenna.png")
image2 = cv2.imread("lenna_noise.png")

# 计算均值哈希
hash1_avg = Mean_hash(image1)
hash2_avg = Mean_hash(image2)

# 计算差值哈希
hash1_dct = Difference_Hash(image1)
hash2_dct = Difference_Hash(image2)

# 计算汉明距离
distance_avg = hamming_distance(hash1_avg, hash2_avg)
distance_dct = hamming_distance(hash1_dct, hash2_dct)

# 打印结果
print(f"Mean Hash Distance: {distance_avg}")
print(f"Difference Hash Distance: {distance_dct}")
