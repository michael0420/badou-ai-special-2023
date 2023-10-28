import cv2
import numpy as np


def color2Gray(img):
    #获取图像大小
    h,w,channel = img.shape

    #创建灰度图模板
    gray = np.zeros((h, w), img.dtype)

    #遍历图像每一个像素
    for i in range(h):
        for j in range(w):

            # 通过公式计算灰度值
            gray[i][j] = int(img[i][j][0]*0.11+img[i][j][1]*0.59+img[i][j][2]*0.3)


    return gray


def gray2Binary(gray_img):

    #拷贝原始灰度图像
    bin_img = gray_img.copy()

    # print(bin_img)

    #获取灰度图像长宽
    h,w = bin_img.shape

    #二值化灰度图像
    for i in range(h):
        for j in range(w):
            if bin_img[i][j] > 127:
                bin_img[i][j] = 255

            else:
                bin_img[i][j] = 0


    # print(bin_img)

    return bin_img


#获取源图像
img = cv2.imread('lenna.jpg')

#测试生成灰度图
gray_img = color2Gray(img)

#测试生成二值化图
bin_img = gray2Binary(gray_img)

#保存图像
cv2.imwrite("gray.jpg",gray_img)
cv2.imwrite("binary.jpg",bin_img)
