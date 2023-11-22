import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets #导入方法类
load_iris = datasets.load_iris() #加载iris数据集
X = load_iris.data #加载特征数据
# y = load_iris.target #加载标签数据
pca = PCA(n_components=2)   #设置降到2维
newX=pca.fit_transform(X)   #训练并得到降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
# print(pca.explained_variance_ratio_)  #输出贡献率
print('原数据x',X)                  #输出原数据
print('sklearn降维数据newx\n',newX)                  #输出降维后的数据
