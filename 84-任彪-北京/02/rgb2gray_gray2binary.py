import cv2
import numpy as np


def change_gray(img ):
    # 首先获取到图片的每个像素点
    img_x, img_y = img.shape[:2];
    # img_x 是图片的行数  img_y 是图片的列数
    #定义一个空的图片,像素结合，和像素点数据的数据类型
    img_gray = np.zeros([img_x, img_y], img.dtype)
    #循环每个像素点
    #填充所有的行
    for i in range(img_x):
        #填充所有的列坐标
        for j in range(img_y):
            # 首先获取图片的当前像素点的bgr坐标,因为彩色图片默认是3通道，所以该n 是 一个三维数据
            n = img[i, j]
            # 然后获取该像素点的灰度值
            # 因为采用cv2 读取的bgr ，并不是rgb，
            # 采用公式 gray = 0.3R +0.59G + 0.11B  科学界试验出来的
            grayPoint = 0.3 * n[2] + 0.39 * n[1] + 0.11 * n[0]
            # 因为像素都是整数，需要取整
            img_gray[i, j] = int(grayPoint);
    return img_gray


def change_binary(img):
    img_h, img_w = img.shape[:2]
    img_binary = np.zeros([img_h, img_w])
    for i in range(img_h):
        for j in range(img_w):
            point = 0
            if (img[i, j] < 128):
                point = 0
            else:
                point = 1
            img_binary[i, j] = point

    return img_binary

if __name__ == '__main__':
    img = cv2.imread('/Users/xiaoluoluosuper-/Downloads/lenna.png')
    new_img = change_gray(img)
    cv2.imshow('gray image', new_img)
    img_binary = change_binary(new_img)
    cv2.imshow('binary img', img_binary)

    cv2.waitKey()