import cv2
import numpy as np

"""
@author: BraHitYQ
最邻近插值法
"""

"""
代码功能解析：
    1、获取输入图像的高度、宽度和通道数。
    2、创建一个空的800x800x通道数的numpy数组，用于存储缩放后的图像。
    3、计算高度和宽度在800中的缩放比例。
    4、遍历800x800的每个像素点，根据缩放比例计算出原图像中对应的位置，并将该位置的像素值赋给新图像的对应位置。
    5、返回缩放后的图像。
"""

def function(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = 800 / height
    sw = 800 / width
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)  # int(),转为整型，使用向下取整。
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


# cv2.resize(img, (800,800,c),near/bin)

img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)


