import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像灰度颜色
img = cv2.imread('lenna.png',0)
print(img.shape)

#获取图片的宽高
rows, cols = img.shape[:]

#图像二维像素降维成一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

#设置停止条件(type, max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means 聚类，聚类成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

#生成最终图像
dst = labels.reshape((img.shape[0],img.shape[1]))
print('src',img)

print('dst',dst)

#显示图像
titles = ['原始图像', '聚类图像']
images = [img,dst]

for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()