# 椒盐噪声
import random
import cv2


def impulse(imgSrc, ratio):
    retImg = imgSrc
    # 通过比率计算需要制造椒盐噪声的矩阵数
    NoiseNum = int(ratio * imgSrc.shape[0] * imgSrc.shape[1])
    for i in range(NoiseNum):
        # 随机出点的位置
        randX = random.randint(0, imgSrc.shape[0] - 1)
        randY = random.randint(0, imgSrc.shape[1] - 1)
        # 这里黑与白的几率各自一半，所以这里取大于与小于0.5
        if random.random() <= 0.5:
            retImg[randX, randY] = 0
        else:
            retImg[randX, randY] = 255
    return retImg


if __name__ == "__main__":
    # 这里需要设置flags为0
    img = cv2.imread("../lenna.png", 0)
    gaussianImg = impulse(img, 0.1)
    img = cv2.imread("../lenna.png")
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('old.png', img2)
    cv2.imshow('impulse.png', gaussianImg)
    cv2.waitKey(0)