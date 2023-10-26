# 灰度化
import cv2
import numpy


def parse_gray_vale(img):
    # 获取图片长宽
    high, wide = img.shape[:2]
    # 绘制新的矩阵
    new_img = numpy.zeros([high, wide], img.dtype)
    print(type(new_img))
    for i in range(high):
        for j in range(wide):
            # 拿出原来的值重新计算得到对应的gray值
            tmp = img[i, j]
            new_img[i, j] = int(tmp[0] * 0.11 + tmp[1] * 0.59 + tmp[2] * 0.3)
    print(new_img)
    # 打印出对比图
    # cv2.imshow("show img gray", new_img)


if __name__ == '__main__':
    # 读取文件信息，以BGR的方式
    img = cv2.imread("lenna.png")
    print(type(img))
    parse_gray_vale(img)