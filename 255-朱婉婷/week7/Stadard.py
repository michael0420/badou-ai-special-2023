
"""
实现归一化的三种方式
"""
import numpy as np
import matplotlib.pyplot as plt
import pylab

#最大最小值归一化1（Min-Max Scaling）
def Minmax_scaling1(X):
    return (X - min(X)) / (max(X) - min(X))

#最大最小值归一化2（Min-Max Scaling）
def Minmax_scaling2(X):
    return (X-np.mean(X))/(max(X) - min(X))

#Z-score标准化（Standardization）
def z_score(X):
    return  (X- np.mean(X)) / np.std(X)

X =np.array([-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
)

"""
sort_idxs= np.argsort(X)
X_sort = X[sort_idxs]

pylab.plot(X_sort,Minmax_scaling1(X_sort),label= "Min-Max Scaling of Min")
pylab.plot(X_sort,Minmax_scaling2(X_sort),label ='Min-Max Scaling of Mean')
pylab.plot(X_sort,z_score(X_sort),label = 'Z_Score')
pylab.legend()
pylab.show()"""

plt.plot(Minmax_scaling1(X))
plt.plot(Minmax_scaling2(X))
plt.plot(z_score(X))
plt.show()