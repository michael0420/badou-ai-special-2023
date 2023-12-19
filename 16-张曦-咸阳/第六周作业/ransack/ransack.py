import numpy as np
import scipy as sp
import scipy.linalg as sl


class LinearLeastSquareModel:
    def __init__(self, input_columns_x, output_columns_y):
        self.input_columns_x = input_columns_x
        self.output_columns_y = output_columns_y

    def fit(self, data):
        columns_x = np.array(
            [data[:, i] for i in self.input_columns_x]).T  # 因为输入数据是类似 [[x,y], [x,y], [x,y]] 的 3行2列矩阵,将x单独做为一列，将y单独作为一列
        columns_y = np.array(
            [data[:, i] for i in self.output_columns_y]).T  # 因为输入数据是类似 [[x,y], [x,y], [x,y]] 的 3行2列矩阵,将x单独做为一列，将y单独作为一列

        """
        scipy.linalg.lstsq 函数是 SciPy 库中用于执行最小二乘拟合（Least Squares Fit）的工具。
        最小二乘法是一种用于拟合线性模型的优化方法，通常用于找到使观测数据与模型预测之间残差平方和最小化的参数。
        
        该函数返回一个包含以下元素的元组 (x, resids, rank, s)：
        x: 拟合的参数向量。
        resids: 残差的平方和。
        rank: 系数矩阵的秩。
        s: 系数矩阵的奇异值。
        """
        x, resids, rank, s = sl.lstsq(columns_x, columns_y)  # residues:残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        columns_x = np.vstack([data[:, i] for i in self.input_columns_x]).T  # 因为输入数据是类似 [[x,y], [x,y], [x,y]] 的 3行2列矩阵,将x单独做为一列，将y单独作为一列
        columns_y = np.vstack([data[:, i] for i in self.output_columns_y]).T  # 因为输入数据是类似 [[x,y], [x,y], [x,y]] 的 3行2列矩阵,将x单独做为一列，将y单独作为一列
        B_fit = sp.dot(columns_x, model)
        error_per_point = np.sum((columns_y - B_fit) ** 2,axis=1)
        return error_per_point


def random_partation(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idx = np.arange(n_data)
    np.random.shuffle(all_idx)
    idx1 = all_idx[:n]
    idx2 = all_idx[n:]
    return idx1, idx2


def ransac(data, model, n, k, t, d):
    iterations = 0
    bestfit = None
    best_err = np.inf  # 设置默认值
    best_inliner_idxs = None
    while iterations < k:
        test_idx, other_idx = random_partation(n, data.shape[0])  # 随机获取测试点，获取随机测试点以外的点 所在行
        maybe_inliners = data[test_idx, :]  # 可能是内点的点的集合
        other_points = data[other_idx]      # 可能是内点以外的点的集合
        maybemodel = model.fit(maybe_inliners)  # 随机选取的inliner的点进行拟合
        test_err = model.get_error(other_points, maybemodel)  # 用可能是模型的这个model 对其他外点计算平方差 # 计算误差:平方和最小
        also_idxs = other_idx[test_err < t]  # 用这个可能是拟合的model计算出来的外点的误差,和阈值t进行比价
        also_inliners = data[also_idxs, :]

        if len(also_inliners) > d:  # 记下内群数量，和给定的条件阈值进行比较，如果这些可能满足模型内的点数量大于 给定误差d,则说明这个模型是有效的
            better_data = np.concatenate((maybe_inliners, also_inliners))  # 样本连接
            better_model = model.fit(better_data)  # 计算满足要求点的最终的模型（model）
            better_errs = model.get_error(better_data, better_model)  # 根据model计算所有点的误差
            this_err = np.mean(better_errs)  # 平均误差作为新的误差
            if this_err < best_err:
                bestfit = better_model
                best_err = this_err
                best_inliner_idxs = np.concatenate((test_idx, also_idxs))  # 所有是组成模型的计算的点

        iterations += 1

    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    else:
        return bestfit, {'inliers': best_inliner_idxs}


def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    B_exact = sp.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300)

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == "__main__":
    test()
