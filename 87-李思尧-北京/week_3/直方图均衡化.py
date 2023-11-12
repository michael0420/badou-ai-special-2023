import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram_img():
    # 获取灰度图像
    img = cv2.imread("lenna.png", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 灰度图像的直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.figure()  # 新建一个图像
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")  # X轴标签
    plt.ylabel("# of Pixels")  # Y轴标签
    plt.plot(hist)
    plt.xlim([0, 256])  # 设置x坐标轴范围
    plt.show()

    return gray


def equalization(gray):
    # 灰度图像直方图均衡化
    dst = cv2.equalizeHist(gray)

    # 均衡化后的直方图
    hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
    plt.figure()
    plt.hist(dst.ravel(), 256)
    plt.show()

    cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
    cv2.waitKey()

    pass


if __name__ == '__main__':
    gray = histogram_img()
    equalization(gray)

    pass
