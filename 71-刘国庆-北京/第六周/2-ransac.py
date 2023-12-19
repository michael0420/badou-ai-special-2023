import numpy as np
import scipy as sp
import scipy.linalg as sl
import pylab


# RANSAC算法主函数，用于拟合模型
def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    # 初始化迭代次数、最佳拟合模型、最佳误差、最佳内点索引
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    # RANSAC主循环，执行k次迭代
    while iterations < k:
        # 随机划分数据点为可能内点和测试点
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        # 打印测试点的索引
        print('test_idxs = ', test_idxs)
        # 获取size(maybe_idxs)行数据(Xi,Yi)
        maybe_inliers = data[maybe_idxs, :]
        # 若干行(Xi,Yi)数据点
        test_points = data[test_idxs]
        # 拟合模型
        maybemodel = model.fit(maybe_inliers)
        # 计算误差:平方和最小
        test_err = model.get_error(test_points, maybemodel)
        # 打印测试误差是否小于阈值
        print('test_err = ', test_err < t)
        # 找到测试误差小于阈值的索引，并获取也是内点的数据
        also_idxs = test_idxs[test_err < t]
        print('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs, :]
        # 如果 debug 为 True，则执行以下调试输出
        if debug:
            # 打印测试误差的最小值
            print('test_err.min()', test_err.min())
            # 打印测试误差的最大值
            print('test_err.max()', test_err.max())
            # 打印测试误差的平均值
            print('numpy.mean(test_err)', np.mean(test_err))
            # 打印迭代次数和当前内点的数量
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        # 打印最小数据点数目 d
        print('d = ', d)
        # 如果也是内点的数量大于最小数据点数目，则执行以下操作
        if len(also_inliers) > d:
            # 样本连接
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            # 使用更好的数据重新拟合模型
            bettermodel = model.fit(betterdata)
            # 计算新数据相对于新模型的误差
            better_errs = model.get_error(betterdata, bettermodel)
            # 平均误差作为新的误差
            thiserr = np.mean(better_errs)
            # 如果新的平均误差小于当前最佳误差，则更新最佳拟合模型、最佳误差和最佳内点索引
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1
    # 如果最佳拟合模型仍为 None，抛出值错误
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    # 如果 return_all 为 True，返回最佳拟合模型和最佳内点索引；否则，只返回最佳拟合模型
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    """返回数据的n个随机行和其他 len(data) - n 行"""
    # 获取n_data下标索引
    all_idxs = np.arange(n_data)
    # 打乱下标索引
    np.random.shuffle(all_idxs)
    # 获取前n个索引
    idxs1 = all_idxs[:n]
    # 获取剩余的索引
    idxs2 = all_idxs[n:]
    # 返回前n个索引和剩余的索引
    return idxs1, idxs2


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        # 初始化输入列的索引列表
        self.input_columns = input_columns
        # 初始化输出列的索引列表
        self.output_columns = output_columns
        # 初始化是否启用调试模式的标志
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        # 使用输入列创建矩阵A，转置以使每一列对应一个样本的特征
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        # 使用输出列创建矩阵B，转置以使每一列对应一个样本的输出
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 使用最小二乘法拟合线性模型，得到系数矩阵x，残差和resids，秩rank和奇异值s
        x, resids, rank, s = sl.lstsq(A, B)
        # 返回最小平方和解的向量
        return x

    def get_error(self, data, model):
        # 使用输入列创建矩阵A，转置以使每一列对应一个样本的特征
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        # 使用输出列创建矩阵B，转置以使每一列对应一个样本的输出
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 计算由模型预测的输出值，B_fit = model.k * A + model.b（假设模型是线性的）
        B_fit = sp.dot(A, model)
        # 计算每个样本的平方误差之和
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        # 返回每个样本的误差列表
        return err_per_point


def test():
    # 生成理想数据
    # 样本个数
    n_samples = 500
    # 输入变量个数
    n_inputs = 1
    # 输出变量个数
    n_outputs = 1
    # 随机生成0-20之间的500个数据:行向量
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    # 随机线性度，即随机生成一个斜率
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    # y = x * k
    B_exact = sp.dot(A_exact, perfect_fit)

    # 加入高斯噪声,最小二乘能很好的处理
    # 500 * 1行向量,代表Xi
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    # 500 * 1行向量,代表Yi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    if 1:
        # 添加"局外点"
        n_outliers = 100
        # 获取索引0-499
        all_idxs = np.arange(A_noisy.shape[0])
        # 将all_idxs打乱
        np.random.shuffle(all_idxs)
        # 100个0-500的随机局外点
        outlier_idxs = all_idxs[:n_outliers]
        # 加入噪声和局外点的Xi
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
        # 加入噪声和局外点的Yi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    # setup model
    # 形式([Xi,Yi]....) shape:(500,2)500行2列
    all_data = np.hstack((A_noisy, B_noisy))
    # 数组的第一列x:0
    input_columns = range(n_inputs)
    # 数组最后一列y:1
    output_columns = [n_inputs + i for i in range(n_outputs)]
    debug = False
    # 类的实例化:用最小二乘生成已知模型
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)
    # 使用最小二乘法拟合线性模型
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])
    # 运行RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    if 1:
        # 对 A_exact 数组的第一列进行排序，返回排序后的索引数组 sort_idxs
        sort_idxs = np.argsort(A_exact[:, 0])
        # 使用排序后的索引数组对 A_exact 进行重新排序，生成秩为2的数组 A_col0_sorted
        A_col0_sorted = A_exact[sort_idxs]
        # 绘制原始数据散点图
        pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
        # 绘制RANSAC拟合的数据散点图
        pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    else:
        # 绘制无异常点的数据散点图
        pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
        # 绘制异常点的数据散点图
        pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')
    # 绘制RANSAC拟合曲线
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
    # 绘制理想系统曲线
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='exact system')
    # 绘制最小二乘法线性拟合曲线
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label='linear fit')
    # 添加图例
    pylab.legend()
    # 显示图形
    pylab.show()


if __name__ == "__main__":
    test()
