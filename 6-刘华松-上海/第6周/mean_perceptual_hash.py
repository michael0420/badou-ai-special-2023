import cv2
import numpy as np

#均值哈希（Mean Hash）是一种基于颜色的哈希算法，用于比较图像的相似度。
#它的原理是将图像缩小为固定大小，计算所有像素的平均值，然后根据像素值的大小将每个像素转为0或1。
#插值哈希（Perceptual Hash）也是一种用于图像相似度比较的算法，
#它首先将图像进行灰度化处理，然后通过计算灰度图像的差异来生成哈希值。

def mean_hash(image, size=8):
    # 缩放图像为指定大小
    image = cv2.resize(image, (size, size))
    # 灰度化图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算像素平均值
    mean = np.mean(gray)
    # 将像素按平均值大小转为0或1
    hash_value = (gray > mean).flatten().astype(int)
    return hash_value.tolist()
	
	
def perceptual_hash(image, size=32):
    # 缩放图像为指定大小
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    # 灰度化图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算灰度差异
    diff = cv2.resize(cv2.absdiff(cv2.GaussianBlur(gray, (5, 5), 0), gray), (size, size), interpolation=cv2.INTER_AREA)
    # 将差异值按阈值转为0或1
    hash_value = (diff > np.mean(diff)).flatten().astype(int)
    return hash_value.tolist()
	

image = cv2.imread('lenna.png')
mean_hash_value = mean_hash(image)
perceptual_hash_value = perceptual_hash(image)
print(mean_hash_value)
print(perceptual_hash_value)	