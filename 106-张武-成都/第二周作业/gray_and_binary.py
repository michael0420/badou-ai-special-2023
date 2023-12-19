import cv2 as cv
import numpy as np


# 图像灰度化
def bgr2gray(img):
    # 获取图片宽高
    h, w = img.shape[:2]

    # 创建一个大小一样的全为零的矩阵
    img_gray = np.zeros([h, w], dtype=img.dtype)

    # 遍历像素点
    for i in range(h):
        for j in range(w):
            # 拿到一个像素点的rgb值
            m = img[i, j]
            # 灰度化像素点，R*0.3+G*0.59+B*0.11
            # gray_bonus = int(m[0] * 0.3 + m[1] * 0.59 + m[2] * 0.11)
            gray_bonus = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
            # 填充到新图像
            img_gray[i, j] = gray_bonus

    return img_gray

# 读取图片
img = cv.imread('lenna.png')

# 灰度化 - 自己实现
img_gray = bgr2gray(img)
cv.imshow('灰度图', img_gray)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('灰度图.png', img_gray)

# 灰度化 - 调用cv
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite('灰度图2.png', img_gray)

# 读取灰度图
img_gray = cv.imread('灰度图.png', cv.IMREAD_GRAYSCALE)

# 二值化
def binary(img_gray):
    # 获取图像宽高
    h, w = img_gray.shape[:2]

    # 创建同样大小的二值图容器
    img_binary = np.zeros([h, w], dtype=img_gray.dtype)
    for i in range(h):
        for j in range(w):
            m = img_gray[i, j]
            if m / 255 <= 0.5:
                img_binary[i, j] = 0
            else:
                img_binary[i, j] = 255
    return img_binary

img_binary = binary(img_gray)
cv.imwrite('二值化图.png', img_binary)
cv.imshow('二值化', img_binary)
cv.waitKey(0)

# cv 二值化
ret, img_binary = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
print(ret)
print(img_binary)
cv.imwrite('二值化图_cv.png', img_binary)