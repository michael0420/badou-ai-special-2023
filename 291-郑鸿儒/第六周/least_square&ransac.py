# 加上偏移量后效果不是很好，暂时没找到原因。。。
import numpy as np
import scipy as sp
import scipy.linalg as sl
import pylab


def random_partition(n, data):
    # 取所有下标
    idx_all = np.arange(data)
    # 打乱索引顺序
    np.random.shuffle(idx_all)
    # 取n个样本作为内群
    idx_1 = idx_all[:n]
    # 余下作为测试集，评价本次取值好坏
    idx_2 = idx_all[n:]
    return idx_1, idx_2


def ransac(data, model, n, k, t, d):
    """
    :param data: 总数据
    :param t: 最大循环次数
    :param d: 可视作内群的最小内群点数量
    :param n 样本数量
    :param model 数据模型
    :param k 可以视作内群的底线
    :return:
    """
    # print(data, t, d, n, model, k)
    iterations = 0
    best_error = np.inf
    best_fit = None
    max = 0

    while k > iterations:
        # print(111)
        current_inlier_idx, text_idx = random_partition(n, data.shape[0])
        current_inlier = data[current_inlier_idx, :]
        test_inlier = data[text_idx, :]
        # print(data)
        # print(current_inlier)
        current_params = model.fit(current_inlier)
        test_error = model.get_error(test_inlier, current_params)
        also_inlier_idx = text_idx[test_error < t]
        also_inlier = data[also_inlier_idx, :]

        # print(len(also_inlier))
        if len(also_inlier) > max:
            max = len(also_inlier)

        if len(also_inlier) > d:
            better_data = np.concatenate((current_inlier, also_inlier))
            better_model = model.fit(better_data)
            better_error = model.get_error(better_data, better_model)
            current_error = np.mean(better_error)
            if current_error < best_error:
                best_fit = better_model
                best_error = current_error
                best_inlier_idx = np.concatenate((current_inlier_idx, also_inlier_idx))
        iterations += 1
    print(max, d)
    if best_fit is None:
        raise ValueError('当前内群阈值下无复合条件模型')
    return best_fit, best_inlier_idx


class LeastSquare:
    def __init__(self, input_column, output_column):
        self.input_column = input_column
        self.output_column = output_column

    def fit(self, data):
        x_pred = np.array([data[:, i] for i in self.input_column]).T
        y_true = np.array([data[:, i] for i in self.input_column]).T

        # 补一列1作为b的系数，以求得b
        params, resids, rank, s = sl.lstsq(np.hstack((x_pred, np.ones((x_pred.shape[0], 1)))), y_true)
        return params

    def get_error(self, data, params):
        x_pred = np.array([data[:, i] for i in self.input_column]).T
        y_true = np.array([data[:, i] for i in self.input_column]).T
        b, k = params
        # print(x_pred, k, b)
        y_pred = sp.dot(x_pred, k) + b
        point_error = np.sum((y_pred - y_true) ** 2, axis=1)
        return point_error


def run_test():
    sample_num = 500
    input_num = 1
    output_num = 1
    x_exact = 20 * np.random.random((sample_num, input_num))
    k_exact = 30 * np.random.normal(size=(input_num, output_num))
    b_exact = 60 * np.random.normal()
    y_exact = np.dot(x_exact, k_exact) + b_exact

    x_noisy = x_exact + np.random.normal(size=x_exact.shape)
    y_noisy = y_exact + np.random.normal(size=y_exact.shape)

    # 设置离群数据个数
    outlier = 100
    all_idx = np.arange(x_noisy.shape[0])
    np.random.shuffle(all_idx)
    outlier_idx = all_idx[:outlier]
    x_noisy[outlier_idx] = 40 * np.random.random((outlier, input_num))
    y_noisy[outlier_idx] = 50 * np.random.normal(size=(outlier, input_num))

    all_data = np.hstack((x_noisy, y_noisy))
    input_column = range(input_num)
    output_column = [input_num + i for i in input_column]

    # print(111,all_data.shape)
    model = LeastSquare(input_column, output_column)
    ransac_fit, ransac_idx = ransac(all_data, model, 50, 100000, 6e4, 270)
    inp = np.array(all_data[:, input_column])
    outp = np.array(all_data[:, output_column])
    res = np.hstack((inp, np.ones((inp.shape[0], 1))))
    linear_fit, resids, rank, s = sl.lstsq(res, outp)

    # 画图
    sort_idx = np.argsort(x_exact[:, 0])
    x_sorted = x_exact[sort_idx]
    pylab.plot(x_noisy[:, 0], y_noisy[:, 0], 'k.', label='data')
    pylab.plot(x_noisy[ransac_idx, :], y_noisy[ransac_idx, :], 'bx', label='RANSAC data')

    k, b = ransac_fit
    b_linear, k_linear = linear_fit
    # print()

    pylab.plot(x_sorted[:, 0], np.dot(x_sorted, k) + b, label="RANSAC fit")
    pylab.plot(x_sorted[:, 0], np.dot(x_sorted, k_exact) + b_exact, label="exact system")
    pylab.plot(x_sorted[:, 0], np.dot(x_sorted, k_linear) + b_linear, label="linear fit")
    pylab.legend()
    pylab.show()


if "__main__" == __name__:
    run_test()
