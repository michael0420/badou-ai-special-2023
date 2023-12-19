# 导入OpenCV库，用于图像处理
import cv2
# 导入NumPy库，并用别名np来使用它。NumPy是一个用于科学计算的库。
import numpy as np
# 导入Matplotlib库，用于绘图
import matplotlib.pyplot as plt

"""
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
"""

# 读取名为'lenna.png'的图像，以灰度模式加载，将加载的图像存储在变量img中
img = cv2.imread("lenna.png", 0)
# 打印图像的形状，即高度和宽度
print(f"图像形状(高度、宽度):\n{img.shape}")
# 获取图像的高度和宽度，并分别存储在rows和cols变量中
rows, cols = img.shape[:]
# 将图像的二维像素数据转换为一维数组，并存储在data中
data = img.reshape(rows * cols, 1)
# 将数据类型转换为32位浮点数
data = np.float32(data)
# 设置K-Means算法的停止条件，包括最大迭代次数（10次）和精确度（1.0）
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 设置K-Means算法的标签为随机中心
flags = cv2.KMEANS_RANDOM_CENTERS
# 使用K-Means算法将图像数据聚类成4类，返回压缩度compactness、标签labels和聚类中心centers
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
# 将标签数据重新变形为与原始图像相同的形状，以得到最终的聚类图像
dst = labels.reshape(img.shape[0], img.shape[1])
# 设置Matplotlib的字体为中文（宋体）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 定义图像的标题
titles = [u'原始图像', u'聚类图像']
# 创建一个包含原始图像和聚类图像的列表
images = [img, dst]
# 对于两个图像执行以下步骤
for i in range(2):
    # 创建一个子图，显示图像，并使用灰度颜色映射
    plt.subplot(1, 2, i + 1)
    plt.imshow(images[i], 'gray')
    # 设置子图的标题
    plt.title(titles[i])
    # 隐藏刻度
    plt.xticks([]), plt.yticks([])
# 显示绘制的图像
plt.show()
