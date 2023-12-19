import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
    ]

clf = KMeans(n_clusters=3)
print('clf:{}'.format(clf))
k_result = clf.fit_predict(X)
print('k_result = {}'.format(k_result))

x = [n[0] for n in X]
print('x:{}'.format(x))
y = [n[1] for n in X]
print('y:{}'.format(y))
plt.scatter(x, y, c=k_result, marker='x')
plt.title('result')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['A, B, C'])
plt.show()


#读取原始图像灰度颜色
# img = cv2.imread('lenna.png',0)
# print(img.shape)
#
# #获取图像高度、宽度
# h, w = img.shape[:]
#
# #图像二维像素转换为一维
# arry = img.reshape((h * w, 1))
# data = np.float32(arry)
#
# #停止条件 (type,max_iter,epsilon)
# cond = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#
# #设置标签
# flag = cv2.KMEANS_RANDOM_CENTERS
#
# #K-Means聚类 聚集成4类
# comp, labels, centers = cv2.kmeans(data, 4, None, cond,10, flag)
#
# #生成最终图像
# dst = labels.reshape((img.shape[0], img.shape[1]))
#
# #用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = "SimHei"
#
#
# #显示图像
# titles = [u'原始图像', u'聚类图像']
# images = [img, dst]
# for i in range(2):
#    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
# plt.show(block=True)
#
