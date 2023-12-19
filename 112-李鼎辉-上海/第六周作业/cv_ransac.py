import numpy as np
import matplotlib.pyplot as plt

'''RANSAC 算法'''
def generate_data(num_points, inlier_ratio=0.8, noise_std=5):
    # 生成一些具有噪声的数据点
    inliers = int(num_points * inlier_ratio)
    outliers = num_points - inliers

    # 生成内点数据
    inlier_x = np.random.rand(inliers) * 100
    inlier_y = 2 * inlier_x + 30 + np.random.normal(0, noise_std, inliers)

    # 生成异常值数据
    outlier_x = np.random.rand(outliers) * 100
    outlier_y = np.random.rand(outliers) * 100

    # 合并内点和异常值
    x = np.concatenate((inlier_x, outlier_x))
    y = np.concatenate((inlier_y, outlier_y))

    # 随机打乱数据
    indices = np.arange(num_points)
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    return x, y


def fit_line_ransac(x, y, num_iterations=100, threshold=5):
    best_inliers = None
    best_line = None

    for _ in range(num_iterations):
        # 随机选择两个点作为内点估计直线
        random_indices = np.random.choice(len(x), size=2, replace=False)
        x_sample = x[random_indices]
        y_sample = y[random_indices]

        # 计算直线参数
        A = np.vstack([x_sample, np.ones_like(x_sample)]).T
        m, c = np.linalg.lstsq(A, y_sample, rcond=None)[0]

        # 计算所有点到估计直线的距离
        distances = np.abs(m * x + c - y)

        # 计算当前内点
        inliers = (distances < threshold)

        # 如果当前内点数多于最好结果，则更新
        if best_inliers is None or np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_line = (m, c)

    return best_line, best_inliers


def plot_ransac_fit(x, y, line_params, inliers):
    plt.scatter(x, y, label='Data points', color='b')

    # 绘制内点
    plt.scatter(x[inliers], y[inliers], label='Inliers', color='r')

    # 绘制RANSAC估计的直线
    m, c = line_params
    plt.plot(x, m * x + c, label='RANSAC Fit', color='g')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('RANSAC Line Fitting')
    plt.show()


# 生成数据
num_points = 100
x, y = generate_data(num_points)

# 使用RANSAC拟合直线
line_params, inliers = fit_line_ransac(x, y)

# 绘制结果
plot_ransac_fit(x, y, line_params, inliers)

