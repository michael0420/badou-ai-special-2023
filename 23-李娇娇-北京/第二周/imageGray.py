import cv2
import numpy as np
import matplotlib.pyplot as plt


def gray(img):
    # 读取原始图像的宽和高
    height, weight = img.shape[: 2]
    # 根据原始图像大小创建一个大小为[height,weight]的矩阵
    img2Grap = np.zeros((height, weight), img.dtype)
    # 灰度化
    for i in range(height):
        for j in range(weight):
            # 将BGR图像转换成灰度图像
            img2Grap[i, j] = int(img[i, j][0] * 0.11 + (img[i, j][1] * 0.59) + (img[i, j][2] * 0.3))
    # print(img2Grap)
    return img2Grap


def binary(img2Gray):
    # 读取原始图像的宽和高
    height, weight = img.shape[: 2]
    # 根据原始图像大小创建一个大小为[height,weight]的矩阵
    img2Binary = np.zeros((height, weight), img.dtype)
    # 二值化
    for i in range(height):
        for j in range(weight):
            if img2Gray[i, j] < 0.5:
                img2Binary[i, j] = 0
            else:
                img2Binary[i, j] = 1
    # print(img2Binary)
    return img2Binary


if __name__ == '__main__':
    # img = cv2.imread("lenna.png")
    # img2Gray = gray(img)
    # cv2.imshow('img', img)
    # cv2.imshow('img2Grap', img2Gray)
    #
    # img2Binary = binary(img2Gray)
    # cv2.imshow('binary', img2Binary)

    plt.subplot(331)
    img = plt.imread("lenna.png")
    plt.imshow(img)
    print("---image lenna----")
    print(img)

    img2Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(332)
    plt.imshow(img2Gray, cmap='gray')
    print("---image gray----")
    print(img2Gray)

    img2Binary = binary(img2Gray)
    plt.subplot(333)
    plt.imshow(img2Binary, cmap='binary')
    print("---image binary----")
    print(img2Binary)

    plt.show()