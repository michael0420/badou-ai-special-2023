import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.linalg as sl

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    :param data: 样本点
    :param model: 模型
    :param n: 生成模型所需的最少样本点
    :param k: 最大迭代次数
    :param t: 阈值:作为判断满足模型的条件
    :param d: 拟合较好时需要的样本点最少的个数
    :param debug:
    :param return_all:
    :return:
    """
    bestfit = None
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    for i in range(k):
        # 随机在数据中选出n个点去求解模型
        n_sample_idxs, test_idxs = random_partition(n, data)
        print(n_sample_idxs, test_idxs)
        n_sample_points = data[n_sample_idxs]
        test_points = data[test_idxs]
        print('n个数据集设置为内群', n_sample_points)
        print('其他数据', test_points)
        # 根据内群和计算出一个模型
        sample_model = model.fit(n_sample_points)
        print('拟合模型:', sample_model)
        # 计算误差:平方和最小
        test_err = model.get_error(test_points, sample_model)
        print('误差', test_err)
        print('t', t)
        print('test_err = ', test_err < t)
        also_idxs = test_idxs[test_err < t]
        print('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs, :]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (i, len(also_inliers)))

        print('d = ', d)
        if (len(also_inliers) > d):
            betterdata = np.concatenate((n_sample_points, also_inliers))  # 样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((n_sample_idxs, also_idxs))  # 更新局内点,将新点加入

    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    # 获取n_data下标索引
    idxs = np.arange(n_data.shape[0])
    # 打乱下标索引
    np.random.shuffle(idxs)
    idxs1 = idxs[:n]
    idxs2 = idxs[n:]
    return idxs1, idxs2


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # 按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 计算最小二乘解, 返回: 回归系数、残差平方和、自变量X的秩、X的奇异值
        linear_fit, residuals, rank, s = np.linalg.lstsq(A, B)
        print('计算最小二乘解', linear_fit, residuals, rank, s)
        return linear_fit

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 计算的y值,B_fit = model.k*A + model.b
        B_fit = sp.dot(A, model)
        # 每一行误差的平方和
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


# 数据量
n_samples = 500  # 样本个数
n_inputs = 1  # 输入变量个数
n_outputs = 1  # 输出变量个数

# 随机生成0-20之间的500个数据:行向量 A_exact.shape = (500*1)
A_exact = 20 * np.random.random((n_samples, n_inputs))
# 随机生成一个斜率
perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
# y = x * k
B_exact = np.dot(A_exact, perfect_fit)

# 加入高斯噪声 - 均匀分布在直线两侧
A_noisy = A_exact + np.random.normal(size=A_exact.shape)    # shape = (500,1)
B_noisy = B_exact + np.random.normal(size=B_exact.shape)    # shape = (500,1)

# 添加局外点噪声
n_outliers = 100
# 获取索引 0-499
all_idxs = np.arange(A_noisy.shape[0])
# 打乱
np.random.shuffle(all_idxs)
# 取100个随机局外点
outlier_idxs = all_idxs[:n_outliers]
# 加入随机局外点噪声
A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))
# 水平堆叠 转换成[Xi, Yi]形式
all_data = np.hstack((A_noisy, B_noisy))

input_columns = range(n_inputs)
output_columns = [n_inputs + i for i in range(n_outputs)]

# 计算最小二乘解
linear_fit, residuals, rank, s = np.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

# run RANSAC 算法
debug = True
model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)
ransac_fit, ransac_data = ransac(all_data, model, 100, 1000, 7e3, 300, debug=debug, return_all=True)
print('ransac: ', ransac_fit, ransac_data)

# 升序排序,方便图像查看
sort_idxs = np.argsort(A_exact[:, 0])
A_col0_sorted = A_exact[sort_idxs]

# 画散点图。
plt.plot(A_noisy, B_noisy, 'bo')
# 最小二乘法拟合
plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label='linear fit')
# ransac
plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='ransac fit')
# 正确的线
plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='perfect fit')
plt.legend()
plt.show()
