import cv2
import random


def function(src, percetage):
    noiseImg = src
    noiseNum = int(percetage * src.shape[0] * src.shape[1])  # 算一下噪声的数量
    for i in range(noiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY 代表随机生成的列
        # random.randint 生成随机整数
        # 椒盐噪声图片边缘不处理，故 -1
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        # random.random 生成随机浮点数，随机取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if random.random() <= 0.5:
            noiseImg[randX, randY] = 0
        else:
            noiseImg[randX, randY] = 255
    return noiseImg


img = cv2.imread("lenna.png", 0)  # flags=0 将图像调整为单通道的灰度图像
img1 = function(img, 0.2)  # 0.8是添加椒盐噪声的百分比
img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("source", img2)
cv2.imshow("lenna_PepperandSalt", img1)
cv2.waitKey(0)
