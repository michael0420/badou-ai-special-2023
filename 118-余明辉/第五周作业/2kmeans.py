from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
#球员数据
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
y = clf.fit_predict(X)

print('y = ',y)
x = [k[0] for k in X]
y = [k[1] for k in X]

plt.scatter(x,y,c=y,marker='*')
plt.show()

#lenna
img = cv2.imread('lenna.png',0)
print(img.shape)
x,y =img.shape

s = img.reshape((x*y,1))
s = np.float32(s)

criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(s, 4, None, criteria, 10, flags)

d = labels.reshape((x,y))

plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, d]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()


