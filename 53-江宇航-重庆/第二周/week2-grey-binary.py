import cv2
import numpy as np

print('Week2_practice')


def get_gray_image(imag):
    img_x, img_y = imag.shape[0:2]
    # img_x 和 img_y 接收img.shape[:2]的值，分别表示图像的行数和列数，以便在后续的代码中创建具有相同尺寸的灰度图像
    img_gray = np.zeros((img_x, img_y), imag.dtype)
    # 创建一个新的空白图像img_gray，其尺寸与输入图像相同，数据类型也相同,这个图像将用于存储灰度版本的图像
    for i in range(img_x):
        # 迭代图像的每一行
        for j in range(img_y):
            # 迭代每一行中的每一个像素列
            n = imag[i, j]
            # 获取当前像素位置(i, j)的BGR（Blue, Green, Red）颜色通道的值，存储在变量n中
            gray_pixel = int(0.11 * n[0] + 0.59 * n[1] + 0.3 * n[2])
            # 浮点法转化灰度
            img_gray[i, j] = gray_pixel
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def change_binary_image(image):
    img_x, img_y = image.shape[0:2]
    # img_x 和 img_y 接收img.shape[:2]的值，分别表示图像的行数和列数，以便在后续的代码中创建具有相同尺寸的灰度图像
    img_binary = np.zeros((img_x, img_y), image.dtype)
    # 创建一个新的空白图像img_binary，其尺寸与输入图像相同，数据类型也相同,这个图像将用于存储灰度版本的图像
    for i in range(img_x):
        # 迭代图像的每一行
        for j in range(img_y):
            # 迭代每一行中的每一个像素列
            n = image[i, j]
            # 像素位置
            if n > 128:
                # 128作为阈值，大于128的像素点设置为1，小于128的像素点设置为0
                img_binary[i, j] = 255
            else:
                img_binary[i, j] = 0
    return img_binary


if __name__ == '__main__':
    image = cv2.imread('C:/Users/15082/Desktop/lenna.png')
    grey_img = get_gray_image(image)
    cv2.imshow('grey', grey_img)
    binary_img = change_binary_image(grey_img)
    cv2.imshow('binary', binary_img)
    cv2.waitKey(0)
