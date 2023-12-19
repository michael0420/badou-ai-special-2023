import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('lenna.png')

data = img.reshape((-1, 3))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

compactness2, labels2, centres2 = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
compactness4, labels4, centres4 = cv2.kmeans(data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
compactness8, labels8, centres8 = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
compactness16, labels16, centres16 = cv2.kmeans(data, 16, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
compactness64, labels64, centres64 = cv2.kmeans(data, 64, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centres2 = np.uint8(centres2)
res2 = centres2[labels2.flatten()]
dst2 = res2.reshape(img.shape)

centres4 = np.uint8(centres4)
res4 = centres4[labels4.flatten()]
dst4 = res4.reshape(img.shape)

centres8 = np.uint8(centres8)
res8 = centres8[labels8.flatten()]
dst8 = res8.reshape(img.shape)

centres16 = np.uint8(centres16)
res16 = centres16[labels16.flatten()]
dst16 = res16.reshape(img.shape)

centres64 = np.uint8(centres64)
res64 = centres64[labels64.flatten()]
dst64 = res64.reshape(img.shape)

# 读入为bgr，plt显示需要rgb
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

plt.rcParams['font.sans-serif'] = ['SimHei']
# u'' 使用Unicode编码
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4', u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 K=64']
imgs = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
    # 生成 2 * 3 表格
    plt.subplot(2, 3, i + 1)
    plt.imshow(imgs[i], 'gray')
    plt.title(titles[i])
    # 不显示xy轴刻度
    plt.xticks([]), plt.yticks([])
plt.show()
