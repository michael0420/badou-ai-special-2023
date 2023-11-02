# 高斯噪声
import cv2
import random


def gaussian(imgSrc, means, sigma, ratio):
    retImg = imgSrc
    # 通过比率计算需要制造高斯噪声的矩阵数
    NoiseNum = int(ratio * imgSrc.shape[0] * imgSrc.shape[1])
    for i in range(NoiseNum):
        # 每次随机取一个点，这里不处理图片两边的边缘所以减一
        randX = random.randint(0, imgSrc.shape[0] - 1)
        randY = random.randint(0, imgSrc.shape[1] - 1)
        # 随机的点这里加上随机的高斯值
        retImg[randX, randY] = retImg[randX, randY] + random.gauss(means, sigma)
        retImg[randX, randY] = retImg[randX, randY] + random.gauss(means, sigma)
        # 这里将值限定在0~255之间，如果有超过改为只有黑白值
        if retImg[randX, randY]< 0:
            retImg[randX, randY] = 0
        elif retImg[randX, randY] > 255:
            retImg[randX, randY] = 255
    return retImg


if __name__ == "__main__":
    # 这里需要设置flags为0
    img = cv2.imread("../lenna.png",0)
    gaussianImg = gaussian(img, 2, 4, 0.8)
    img = cv2.imread("../lenna.png")
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('old.png', img2)
    cv2.imshow('gaussian.png', gaussianImg)
    cv2.waitKey(0)

