import cv2
import random


def noise_img(src, percentage):
    NoiseImg = src
    # shape[0]=图像的高，shape[1]=图像的宽，shape[2]=图像的图像通道数量
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 椒盐噪声图片边缘不处理，故-1
        # 噪点定位
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        # random.random生成[0, 1)之间的随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


if __name__ == '__main__':
    img0 = cv2.imread('lenna.png', 0)
    img1 = noise_img(img0, 0.2)
    img = cv2.imread('lenna.png')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('source', img2)
    cv2.imshow('lenna_PepperandSalt', img1)
    cv2.waitKey()

    pass
