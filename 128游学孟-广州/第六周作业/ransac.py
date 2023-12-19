import numpy as np
import matplotlib.pyplot as plt

#生成随机点
np.random.seed(0)
x = np.random.uniform(0,10,100)
y = 2*x+3+np.random.normal(0,10,100)

# 添加离群点
outliners_x = np.random.uniform(0,10,10)
outliners_y = np.random.uniform(0,100,10)
x = np.concatenate((x,outliners_x))
y = np.concatenate((y,outliners_y))

# 使用RANSAC算法拟合直线
"""
:param best_inliers:存储最佳匹配模型得内点索引
:param best_a:最佳匹配模型得斜率参数。
:param best_b:最佳匹配模型得截距参数。
:param max_iterations:算法的最大迭代次数，
:inlier_threshold:内点的阈值。
"""
best_inliers = []
best_a = 0
best_b = 0
max_iterations = 1000
inlier_threshold = 2

for i in range(max_iterations):
    # 随机选择两个点
    idx = np.random.choice(len(x),2,replace = False)
    x_inliers = x[idx]
    y_inliers = y[idx]

    # 计算拟合直线参数（a,b）
    a = (y_inliers[1]-y_inliers[0])/(x_inliers[1]-x_inliers[0])
    b =y_inliers[0] - a*x_inliers[0]

    # 计算其他带你到直线的距离
    distances = np.abs(y-(a*x+b))

    # 判断点是否为内群点
    inliers = distances < inlier_threshold

    # 更新最优模型以及内点数量
    if np.sum(inliers) > len(best_inliers):
        best_inliers = inliers
        best_a = a
        best_b = b


print('Best model : y = {}x + {}'.format(best_a,best_b))
print("Best_inliers",np.sum(best_inliers))

#绘制结果
plt.scatter(x[best_inliers],y[best_inliers],color = 'g',label = 'Inliers')
plt.scatter(x[~best_inliers],y[~best_inliers],color = 'r',label = 'Outliners=')
plt.plot(x,best_a*x+best_b,color = 'b',label = 'Best Fit Line' )
plt.legend()
plt.show()

