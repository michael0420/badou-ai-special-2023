# coding=utf-8

import numpy as np
from sklearn.decomposition import PCA

"""
if you winna learn more about the function, you should press "Ctrl" and click the function, then it will go to the origin code. At there you can read the code hinting which can better help you to understand the function. 
如果你想了解更多关于该函数的信息，你应该按“Ctrl”并单击该函数，然后它将转到原始代码。在那里你可以阅读代码提示，这可以更好地帮助你理解函数。

1. fit的方法(它用于拟合模型。该方法接受两个参数：X和y):
     1.1. X是一个形状为(n_samples, n_features)的数组，表示训练数据。其中，n_samples是样本数量，n_features是特征数量。
     1.2. y是一个可选参数，但在这个方法中被忽略。


2. fit_transform的方法（拟合模型并对输入数据X进行降维。该方法接受两个参数：X和y（默认为None））:
     2.1 方法的返回值是一个新的数组X_new，其形状为(n_samples, n_components)，表示经过降维后的数据。
     2.2 在方法内部，首先调用了self._fit(X)方法来获取U、S和Vt三个矩阵。然后根据self.n_components_的值对U进行切片操作，保留前n个主成分。
     2.3 根据self.whiten的值来决定如何计算X_new。如果self.whiten为True，则使用白化方法，将X乘以V矩阵并除以S矩阵的平方根乘以样本数减1；否则，直接将X乘以V矩阵。
"""

X = np.array([[-1, 2, 66, -1],
              [-2, 6, 58, -1],
              [-3, 8, 45, -2],
              [1, 9, 36, 1],
              [2, 10, 62, 1],
              [3, 5, 83, 2]])  # 导入数据，维度为4
pca = PCA(n_components=2)   # 降到2维
pca.fit(X)                  # 训练
newX = pca.fit_transform(X)   # 降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_)  # 输出贡献率
print(newX)                  # 输出降维后的数据
