#!/usr/bin/env python
# encoding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl


def random_partition(count, total):
    total_idx = np.arange(total)
    np.random.shuffle(total_idx)
    return total_idx[:count], total_idx[count:]


def ransac(data_x, data_y, model, n, k, c, d):
    """
    ransac 求解线性回归
    :param data_x: 全部x数据
    :param data_y: 全部y数据
    :param model: 模型
    :param n: 随机取样数量
    :param k: 最大迭代次数
    :param c: 阈值
    :param d: 最少多少个内群点才可以认为是拟合较好
    :return: 斜率，截距， 内群点
    """
    # print(data_x.shape, data_y.shape)
    assert len(data_x) == len(data_y), "数据有误"
    bestfit = None
    bestinlier = None
    besterror = np.inf
    iteration = 0
    best_idx = None
    while iteration < k:
        # 随机取点作为内群点
        maybe_inlier_idx, other_idx = random_partition(n, len(data_x))
        maybe_inlier_x = data_x[maybe_inlier_idx]
        maybe_inlier_y = data_y[maybe_inlier_idx]
        other_data_x = data_x[other_idx]
        other_data_y = data_y[other_idx]
        # 求得当前斜率与截距
        maybe_inlier_k, maybe_inlier_b = model.fit(maybe_inlier_x, maybe_inlier_y)
        # 求误差(残差平方和)
        error = model.get_error(maybe_inlier_k, maybe_inlier_b, other_data_x, other_data_y)
        also_inlier_idx = other_idx[error < c]
        also_inlier_x = data_x[also_inlier_idx]
        also_inlier_y = data_y[also_inlier_idx]
        if len(also_inlier_x) > d:
            better_x = np.concatenate((maybe_inlier_x, also_inlier_x))
            better_y = np.concatenate((maybe_inlier_y, also_inlier_y))
            better_k, better_b = model.fit(better_x, better_y)
            better_err = model.get_error(better_k, better_b, better_x, better_y)
            this_error = np.mean(better_err)  # 平均误差作为新的误差
            if this_error < besterror:
                besterror = this_error
                bestinlier = np.concatenate((better_x, better_y), axis=1)
                bestfit = [maybe_inlier_k, maybe_inlier_b]
                best_idx = np.concatenate((maybe_inlier_idx, also_inlier_idx))
        iteration += 1
    if bestfit is None:
        raise ValueError('当前内群阈值下无复合条件模型')
    else:
        return bestfit, best_idx, bestinlier


class LeastSquare(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def fit(self, x, y):
        # 合并一个全1的矩阵求截距，原因是y = kx + b <=> dot([k, b], [x, 1])
        x_with_bias = np.hstack((x, np.ones(x.shape)))
        x, resids, rank, sig = sl.lstsq(x_with_bias, y)
        # print("回归系数 (x):", x[0])
        # print("残差 (residuals):", resids)
        # print("矩阵的秩 (rank):", rank)
        # print("奇异值 (singular values):", sig)
        return x.ravel()

    def get_error(self, k, b, x, y):
        # print(111111, k, b, x, y)
        y_pred = np.dot([k, b], np.hstack((x, np.ones(x.shape))).T)
        ri = y - np.reshape(y_pred, y.shape)
        # print(np.sum(ri ** 2, axis=1))
        return np.sum(ri ** 2, axis=1)


if "__main__" == __name__:
    sample_count = 500  # 500个样本点
    outlier_count = 100  # 100个离群点
    input_dim = 1  # 输入维度
    output_dim = 1  # 输出维度
    x_expect = 20 * np.random.random((sample_count, input_dim))  # x 未添加噪声
    k_expect = 60 * np.random.normal(size=(input_dim, output_dim))  # 斜率k
    b_expect = 20 * np.random.normal()  # 截距
    y_expect = np.dot(x_expect, k_expect) + b_expect  # y = kx + b未添加噪声

    # 添加噪声
    x_noise = x_expect + np.random.normal(size=(sample_count, input_dim))
    y_noise = y_expect + np.random.normal(size=(sample_count, output_dim))

    # 外群
    all_idx = np.arange(sample_count)
    # print(all_idx)
    np.random.shuffle(all_idx)  # 打乱顺序，改变原数组
    outlier_idx = all_idx[:outlier_count]
    # print(x_noise.type)
    x_noise[outlier_idx] = 20 * np.random.random(size=(outlier_count, input_dim))
    y_noise[outlier_idx] = 50 * np.random.normal(size=(outlier_count, output_dim))
    model = LeastSquare(input_dim, output_dim)
    least_k, least_b = model.fit(x_noise, y_noise)
    # ransac 得到的k, b以及推测的内群
    ransac_fit, ransac_fit_idx, ransac_inlier = ransac(x_noise, y_noise, model, n=30, k=200000, c=1e4, d=360)
    # 画图
    plt.figure()
    plt.plot(x_noise, np.dot([least_k, least_b], np.hstack((x_noise, np.ones(x_noise.shape))).T), label='linear',
             color="red")
    plt.plot(x_noise, np.dot(x_noise, k_expect) + b_expect, label='expect', color="green")
    plt.plot(x_noise, np.dot(ransac_fit, np.hstack((x_noise, np.ones(x_noise.shape))).T), label='ransac', color="pink")
    # print(ransac_fit, [k_expect, b_expect], [least_k, least_b])
    plt.scatter(x_noise, y_noise, c="black", marker='x')
    plt.scatter(ransac_inlier[:, 0], ransac_inlier[:, 1], c="blue", marker='x')
    plt.legend()
    plt.show()
