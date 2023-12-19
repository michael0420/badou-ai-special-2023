import numpy as np
import scipy as sp
import scipy.linalg as sl


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    interations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None

    while interations < k:
        maybe_point, test_point = random_partition(n, data.shape[0])
        print("test point = ", test_point)
        maybe_inliers = data[maybe_point, :]
        test_point_idxs = data[test_point]
        maybemodel = model.fit(maybe_inliers)
        model_err = model.get_error(test_point_idxs, maybemodel)
        print("model_err =", model_err < t)  # model_err < t 返回一个布尔数组，其中每个元素是对应位置上的 test_err 元素是否小于阈值 t
        filt_idx = test_point[model_err < t]  # 筛选过的点的编号
        print("test_idx =", filt_idx)
        filt_inliers = data[filt_idx, :]

        if debug:
            print('model_err.min()', model_err.min())
            print('model_err.max()', model_err.max())
            print('numpy.mean(model_err)', np.mean(model_err))
            print(f"interations {interations}: len(filt_idx) = {len(filt_idx)}")
        print(f"d = {d}")
        if (len(filt_idx) > d):
            betterdata = np.concatenate((maybe_inliers, filt_inliers))  # 把这些提前选出来的可能在内群的点和筛选过在内群的点的index拼接在一起
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            new_err = np.mean(better_errs)
            if new_err < besterr:
                bestfit = bettermodel
                besterr = new_err
                best_inlier_idxs = np.concatenate((maybe_point, filt_idx))
        interations += 1

    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {"inliers": best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # A：系数矩阵，即线性系统的系数矩阵。
        # B：常数矩阵，即线性系统的右侧常数矩阵。
        # 这个函数的作用是通过最小二乘法，找到一个向量x，使得线性方程组Ax = B中的残差和最小。返回的x就是这个最小二乘解，resids
        # 是残差和，rank是矩阵A的秩，s是奇异值数组,表示A矩阵的奇异值。。
        x, resids, rank, s = sl.lstsq(A, B)
        return x

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = sp.dot(A, model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def test():
    n_sample = 500
    n_input = 1
    n_output = 1
    A_exact = 20 * np.random.random((n_sample, n_input))
    perfect_fit = 60 * np.random.normal(size=(n_input, n_output))  # 生成一个服从正态分布的随机数
    B_exact = sp.dot(A_exact, perfect_fit)  # B_exact = A_exact * perfect_fit

    A_noise = A_exact + np.random.normal(size=A_exact.shape)
    B_noise = B_exact + np.random.normal(size=B_exact.shape)

    if 1:
        n_outliers = 100
        all_idx = np.arange(A_noise.shape[0])
        np.random.shuffle(all_idx)
        outlier_idx = all_idx[:n_outliers]
        A_noise[outlier_idx] = 20 * np.random.random((n_outliers, n_input))
        B_noise[outlier_idx] = 50 * np.random.normal(size=(n_outliers, n_output))

    all_data = np.hstack((A_noise, B_noise))
    input_columns = range(n_input)
    output_columns = [n_input + i for i in range(n_output)]
    model = LinearLeastSquareModel(input_columns, output_columns)

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=True, return_all=True)

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]

        if 1:
            pylab.plot(A_noise[:, 0], B_noise[:, 0], 'k.', label='data')
            pylab.plot(A_noise[ransac_data['inliers'], 0], B_noise[ransac_data['inliers'], 0],
                       color="#ADFF2F", marker="x", linestyle="", label='RANSAC data')
        else:
            pylab.plot(A_noise[non_outlier_idxs, 0], B_noise[non_outlier_idxs, 0], 'k.', label='noise data')
            pylab.plot(A_noise[outlier_idxs, 0], B_noise[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0], color="#000000",
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0], color="#C71585",
                   label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == "__main__":
    test()
