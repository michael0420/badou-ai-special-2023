import numpy as np
import scipy as sp
import scipy.linalg as sl
import pylab


def ransac(data,model,n,k,t,d,dubug = False,return_all = False):
    iter = 0
    ans = None
    besterr =np.inf
    best_inlier_idxs = None
    while iter<k:
        #随机生成n组数据
        midx,tidx=random_part(n,data.shape[0])
        mdata = data[midx,:]
        tdata = data[tidx,:]
        #通过模型拟合
        mmodel = model.fit(mdata)
        #找到内群点
        terr = model.get_error(tdata, mmodel)
        also_idxs = tidx[terr < t]
        also_inliers = data[also_idxs, :]
        #链接内群点重新拟合
        if (len(also_inliers) > d):
            betterdata = np.concatenate( (mdata, also_inliers) ) #样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs) #平均误差作为新的误差
            if thiserr < besterr:
                ans = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (midx, also_idxs) ) #更新局内点,将新点加入
        iter += 1
    if ans is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return ans,{'inliers':best_inlier_idxs}
    else:
        return ans
    if dubug:
        print('hhhh')

def random_part(n,data):
    #生成随机n组数据
    all_idx = np.arange(data)
    np.random.shuffle(all_idx)
    id1 = all_idx[:n]
    id2 = all_idx[n:]
    return id1,id2


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, resids, rank, s = sl.lstsq(A, B)  # residues:残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = np.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point

def test():
    #生成数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    # 添加"局外点"
    n_outliers = 100
    all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
    np.random.shuffle(all_idxs)  # 将all_idxs打乱
    outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
    A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
    B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi

    data = np.hstack( (A_noisy,B_noisy) )
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    #最小二乘直接拟合
    linear_fit, resids, rank, s = sp.linalg.lstsq(data[:, input_columns], data[:, output_columns])
    #初始化模型
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)
    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(data, model, 50, 1000, 7e3, 300, debug, return_all=True)

    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

    if 1:
        pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
        pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
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
if __name__ == '__main__':
    test()