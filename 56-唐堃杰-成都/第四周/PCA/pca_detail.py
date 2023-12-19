"""
使用PCA求样本矩阵的k阶降维矩阵
"""
import numpy as np
# 1. 矩阵中心化（零均值化）
def centralized(org):
    centrOrg = []
    # 这里拿出所有当前样本的项然后计算其特征均值
    mean = np.array([np.mean(attr) for attr in org.T])
    # 中心化，这里减去均值，零点重新生成
    centrOrg = org - mean;
    return centrOrg

# 2.求样本的协方差矩阵
def covariance(centrOrg):
    # 获取样本的样例总数
    sample_size = np.shape(centrOrg)[0]
    # 求样本的协方差矩阵
    cov = np.dot(centrOrg.T, centrOrg) / (sample_size - 1)
    return cov

# 获取k阶的降维矩阵
def reduction(cov, k):
    # 求出协方差矩阵的特征值与特征向量，这里调用numpy的linalg下的eig可以直接获取
    val,vec = np.linalg.eig(cov)
    print("协方差矩阵特征值",val)
    print("协方差矩阵特征向量",vec)
    # 将特征进行降维排序（从大到小方便后续选取最大）
    index = np.argsort(-1 * val)
    # 通过指定的K构建K阶降维矩阵
    UT = [vec[:, index[i]] for i in range(k)]
    U = np.transpose(UT)
    print("%d阶降维转换矩阵U:\n", k, U)
    return U
# 获取样本矩阵的降维矩阵
def Z(org, U):
    # 原始矩阵乘以降维转换矩阵得到降维矩阵
    z = np.dot(org, U)
    print("样本矩阵的降维矩阵Z:\n", z)
    return z

if __name__ == "__main__":
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print("K的值\n", K);
    # 1. 中心化
    centrOrg = centralized(X)
    # 2. 求协方差矩阵
    cov = covariance(centrOrg)
    # 3.获取k阶的降维矩阵（先获取特征向量与特征值）
    U = reduction(cov, K)
    # 4.获取原始矩阵的降维矩阵
    z = Z(X, U)