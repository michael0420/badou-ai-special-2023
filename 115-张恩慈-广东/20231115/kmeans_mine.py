# 作业2：实现Kmeans
# 步骤：
# 1、将多通道图转换为灰度图（单通道）
# 2、获取原图长高信息
# 3、将二位像素值转换为一维数据
# 4、设置Kmeans的停止条件
# 5、设置标签
# 6、使用 cv2 的 kmeans API，返回 聚类紧密度、每个数据点所属的类别标签、聚类中心的坐标
# 7、根据每个数据点所属的类别标签绘制最终生成图
import cv2
import numpy as np

if __name__ == '__main__':
    # 1、
    gray = cv2.imread('lenna.png', 0)

    # 2、
    w, h = gray.shape[:]

    # 3、
    data = gray.reshape((w * h, 1))
    data = np.float32(data)

    # 4、
    stopCondition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 5、
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 6、
    relate, labels, centerXY = cv2.kmeans(data, 10, None, stopCondition, 10, flags)

    # 7、labels 是 每个数据点所属的聚类中心的标签，是一个一维数组，格式为numpy数组
    # 需要将一维数组 先 归一化
    normalized_labels = cv2.normalize(labels, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # 8、归一化后的一维数组转换为二维数组
    normalized_labels = normalized_labels.reshape((w, h))

    # 9、展示kmeans后的结果
    cv2.imshow('kmeans',  normalized_labels)
    cv2.waitKey(0)