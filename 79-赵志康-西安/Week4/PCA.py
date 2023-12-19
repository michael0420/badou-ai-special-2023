import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
pca = PCA(n_components=2) #指定降为到2维
pca.fit(X)                #训练X
newX=pca.fit_transform(X) #输出新的降维后的数据，赋值给newX
print(newX)
"""
fit(X,y=None) 
fit()可以说是scikit-learn中通用的方法，每个需要训练的算法都会有fit()方法，它其实就是算法中的“训练”这一步骤。因为PCA是无监督学习算法，此处y自然等于None。 
fit(X)，表示用数据X来训练PCA模型。函数返回值：调用fit方法的对象本身。比如pca.fit(X)，表示用X对pca这个对象进行训练。 
fit_transform(X) 
用X来训练PCA模型，同时返回降维后的数据。 
newX=pca.fit_transform(X)，newX就是降维后的数据。 
inverse_transform() 
将降维后的数据转换成原始数据，X=pca.inverse_transform(newX) 
transform(X) 
将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
"""