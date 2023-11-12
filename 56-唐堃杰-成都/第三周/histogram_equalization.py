#直方图均衡化
import cv2
import matplotlib.pyplot as plt

def gradation(cur_gray):
    # 灰度图直方图均衡化
    dst_img = cv2.equalizeHist(cur_gray)
    # 生成直方图
    # 参数：
    #   images：图像矩阵
    #   channels: 通道数
    #   mask：掩码
    #   histSize：直方图的大小，一般等于灰度级数
    #   ranges： 横轴范围
    histogram = cv2.calcHist([dst_img], [0], None, [256], [0, 256])

    plt.figure()
    plt.hist(histogram.ravel(), 256)
    plt.show()

def color_histogram(cur_img):
    # 彩色直方图均衡化
    # 分解成长宽与通道数
    (h, w, c) = cv2.split(cur_img)
    # 分别均衡化
    hE = cv2.equalizeHist(h)
    wE = cv2.equalizeHist(w)
    cE = cv2.equalizeHist(c)

    # hist = cv2.calcHist([hE, wE, cE], [3], None, [256], [0, 256])
    # # plt.figure()
    # # plt.hist()
    # cv2.imshow("tmp", hist)
    # 合并通道
    ret = cv2.merge((hE, wE, cE))
    cv2.imshow("color_histogram", ret)

if __name__ == "__main__":
    # 先获取灰度图
    img = cv2.imread("../lenna.png", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 灰度直方图均衡化
    gradation(gray)
    # color_histogram(img)
    cv2.waitKey(0)

