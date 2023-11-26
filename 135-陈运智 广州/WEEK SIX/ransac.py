import numpy as np
import matplotlib.pyplot as plt

# 生成带有噪声的数据集
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(scale=2, size=len(x))

# 添加一些离群点
y[20] += 20
y[80] -= 20

# 将数据集表示为 (x, y) 的坐标对
data = np.column_stack((x, y))

# 定义拟合直线的模型
def fit_line(data):
    x, y = data.T
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

# 定义计算距离的函数
def calculate_distance(data, model):
    m, c = model
    x, y = data.T
    distances = np.abs(m * x + c - y) / np.sqrt(m**2 + 1)
    return distances

# 定义 RANSAC 算法
def ransac(data, iterations, threshold):
    best_model = None
    best_inliers = np.array([])

    for _ in range(iterations):
        # 随机采样
        random_sample = data[np.random.choice(len(data), 2, replace=False)]
        # 拟合模型
        model = fit_line(random_sample)
        # 计算距离
        distances = calculate_distance(data, model)
        # 标记为内点
        inliers = data[distances < threshold]

        # 更新最佳模型
        if len(inliers) > len(best_inliers):
            best_model = fit_line(inliers)
            best_inliers = inliers

    return best_model, best_inliers

# 运行 RANSAC
iterations = 100
threshold = 3
best_model, best_inliers = ransac(data, iterations, threshold)

# 绘制结果
plt.scatter(x, y, label='Data')
plt.plot(x, 2 * x + 1, color='red', linewidth=2, label='Ground Truth')
plt.plot(x, np.polyval(best_model, x), color='green', linewidth=2, label='RANSAC Fit')
plt.scatter(best_inliers[:, 0], best_inliers[:, 1], color='orange', label='Inliers')
plt.legend()
plt.show()
