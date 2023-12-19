"""
实现RANSAC算法
"""
import numpy as np
import scipy as sp
import scipy.linalg as sl

#设置给定模型（最小二乘法模型）
class LinearLeastSquareModel:
    #输入：输入变量和输出变量列索引+不打印调试信息
    def __init__(self, input_columns, output_columns, debug= False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    
    #最小二乘法拟合结果数据
    def fit(self, data):
        A = data[:,self.input_columns]
        B = data[:,self.output_columns]

        #使用sl.lstsq进行最小二乘拟合，得到拟合解
        #输出最小二乘解、残差和、矩阵秩、奇异值分解的奇异值
        #B = Ax
        x, resids, rank, s = sl.lstsq(A,B)
        return x
    
    #计算数据点和拟合模型的误差
    def get_error(self, data, model):
        A = data[:,self.input_columns]
        B = data[:,self.output_columns]
        #计算点积（y值）,B_fit = model.k*A+model.b
        B_fit = np.dot(A,model)
        err_per_point = np.sum((B-B_fit)**2,axis=1)
        return err_per_point
    
#定义随机选取n个点函数
def random_partition(n, data_len):
    #获取索引并打乱顺序
    all_idxs = np.arange(data_len)
    np.random.shuffle(all_idxs)

    #得到随机选取的n个样本和剩余样本
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

    
#定义ransac函数
"""
输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
        debug - 是否打印调试信息
        return_all - 是否返回所有的拟合结果
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）
"""
def ransac(data,model, n, k, t, d, debug=False, return_all=False):
    iterations = 0
    bestfit = None
    besterr = np.inf #默认值
    best_inlier_idxs = None

    #在迭代次数内
    while iterations < k :
        #随机划分n个样本点作为训练点，剩余作为测试
        maybe_idxs, test_idxs = random_partition(n,data.shape[0])
        
        #获取maybe_idxs对应样本点（xi,yi)
        maybe_inliers = data[maybe_idxs, :]
        test_points = data[test_idxs,:]

        #对训练点拟合模型
        maybemodel = model.fit(maybe_inliers)
        #计算样本点和模型的误差
        test_err = model.get_error(test_points,maybemodel)

        #根据误差阈值t筛选符合条件的样本索引
        also_idxs = test_idxs[test_err < t]
        #选取符合条件的样本点
        also_inliers = data[also_idxs,:]

        #如果测试中的符合样本数量大于阈值d
        if (len(also_idxs) > d):
            #将训练集和符合条件的样本点拼在一起
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            #训练模型
            bettermodel = model.fit(betterdata)
            #计算误差
            better_errs = model.get_error(betterdata,bettermodel)
            #计算所有点平均误差
            thiserr = np.mean(better_errs)
            
            #如果新的误差比最优误差小：
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr

                #更新局内点索引
                best_inlier_idxs = np.concatenate((maybe_idxs,also_idxs))
        iterations += 1

        if bestfit is None:
            raise ValueError("didn't meet fit acceptance criteria")
        if return_all:
            return bestfit,{'inliers':best_inlier_idxs}
        else:
            return bestfit
        

def test():
    n_samples = 500 #样本个数
    n_inputs = 1 #输入变量维数
    n_outputs = 1 # 输出变量维数

    #生成随机数据作为输入变量
    #B=A*perfect_fit
    A_exact = 20* np.random.random((n_samples,n_inputs))       
    perfect_fit = 60*np.random.normal(size = (n_inputs,n_outputs))
    #理想输出变量
    B_exact = np.dot(A_exact,perfect_fit)

    #加入高斯噪声
    #注意size啊！！！！！！！
    A_noisy = A_exact + np.random.normal(size = A_exact.shape)
    B_noisy = B_exact + np.random.normal(size = B_exact.shape)

    #添加局外点，默认执行
    if 1:
        n_outliers = 100

        #生成随机局外点索引,100个在0-500之间的点
        all_idxs = np.arange(A_noisy.shape[0])
        np.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]

        #将局外点的输入变量赋值
        A_noisy[outlier_idxs] = 20*np.random.random((n_outliers,n_inputs))
        B_noisy[outlier_idxs] = 50*np.random.normal(size = (n_outliers,n_outputs))

    #将输入变量和输出变量列链接
    all_data = np.hstack((A_noisy, B_noisy))

    #定义输入变量和输出变量的列索引
    input_columns = range(n_inputs)
    output_columns = [n_inputs+i for i in range(n_outputs)]
    debug = False

    #最小二乘拟合生成已知模型
    model = LinearLeastSquareModel(input_columns, output_columns, debug)
    
    #线性最小二乘拟合
    linear_fit, resids, rank , s = sp.linalg.lstsq(all_data[:,input_columns],all_data[:,output_columns])

    #RANSAC算法，得到拟合结构和局内点数据索引
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug, return_all=True)

    if 1:
        import pylab

        #对第一列进行排序
        sort_idxs = np.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sort_idxs]

        if 1:
            pylab.plot(A_noisy[:,0], B_noisy[:,0], 'k.', label='data')
            #绘制局内点数据
            pylab.plot(A_noisy[ransac_data['inliers'],0], B_noisy[ransac_data['inliers'],0], 'bx', label='RANSAC data')
        else:
            #绘制噪声和局外点
            pylab.plot( A_noisy[non_outlier_idxs,0], B_noisy[non_outlier_idxs,0], 'k.', label='noisy data' )
            pylab.plot( A_noisy[outlier_idxs,0], B_noisy[outlier_idxs,0], 'r.', label='outlier data' )
    
        #拟合结果
        pylab.plot(A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,ransac_fit)[:,0],
                    label='RANSAC fit')
        
        #理想系统
        pylab.plot(A_col0_sorted[:,0],
                   np.dot(A_col0_sorted, perfect_fit)[:,0],
                   label='exact system')
        
        #线性拟合结果
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,linear_fit)[:,0],
                    label='linear fit')
        pylab.legend()
        pylab.show()

if __name__ == '__main__':
    test()
        

    