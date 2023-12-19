# 使用不同的k划分图像
import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 读取图片将其转换为一维并将数据变为浮点数
    img = cv2.imread("../lenna.png")
    img_data = img.reshape((-1,3))
    data = np.float32(img_data)
    # 设置停止条件(分别是类型，循环次数与误差)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置初始质心点
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 使用不同的k用k-means聚类
    compactness2, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10,flags)
    compactness4, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
    compactness8, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
    compactness16, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
    compactness64, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

    # 将数据转换为原来的格式
    centers2 = np.uint8(centers2)
    res = centers2[labels2.flatten()]
    dst2 = res.reshape(img.shape)

    centers4 = np.uint8(centers4)
    res = centers4[labels4.flatten()]
    dst4 = res.reshape((img.shape))

    centers8 = np.uint8(centers8)
    res = centers8[labels8.flatten()]
    dst8 = res.reshape((img.shape))

    centers16 = np.uint8(centers16)
    res = centers16[labels16.flatten()]
    dst16 = res.reshape((img.shape))

    centers64 = np.uint8(centers64)
    res = centers64[labels64.flatten()]
    dst64 = res.reshape((img.shape))

    # 将数据转换为RGB显示
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
    dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
    dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
    dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
    dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

    # 设置中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 依次显示图像
    titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
              u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 K=64']
    images = [img, dst2, dst4, dst8, dst16, dst64]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()