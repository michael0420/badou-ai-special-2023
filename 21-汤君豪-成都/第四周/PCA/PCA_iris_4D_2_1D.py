import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

#加载数据，x表示数据集中的属性数据，y表示数据标签
X, Y = load_iris(return_X_y=True)
print(X)
#加载pca算法，设置降维后主成分数目为2
pca = PCA(n_components=1)
pca.fit(X)
#对原始数据进行降维，保存在reduced_x中
reduced_X = pca.fit_transform(X)
rx, ry = [], []
bx, by = [], []
gx, gy = [], []
for i in range(len(reduced_X)):
    if Y[i] == 0:
        rx.append(reduced_X[i][0])
        ry.append(0)
    elif Y[i] == 1:
        bx.append(reduced_X[i][0])
        by.append(0)
    else:
        gx.append(reduced_X[i][0])
        gy.append(0)
plt.scatter(rx, ry, c='r', marker='x')
plt.scatter(bx, by, c='b', marker='D')
plt.scatter(gx, gy, c='g', marker='.')
plt.show()