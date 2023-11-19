# PCA算法：
# 1、对矩阵src 中心化（零均值化），结果为E
# 2、对E求协方差矩阵M
# 3、对协方差矩阵M求特征向量和特征值
# 4、从特征值由大到小排序，取前K个组成 降维转换矩阵
# 5、通过 原矩阵和降维转换矩阵相乘，得到 降维后的矩阵

import numpy as np

# 设置降维后矩阵维度K
K = 2

# 1、中心化
def centralization(src):
    # 1.1 先对src的行列做转置，需要将特征作为列以便后续处理
    mean = np.array([np.mean(attr) for attr in src.T])
    print('样本集的特征值均值的数组：\n',mean)
    # 1.2 中心化
    center = src - mean
    print('样本集中心化后:\n',center)
    return center

# 2、对中心化结果 求 协方差矩阵
def getCova(center):
    # 2.1 求样本的总样本数 行数
    count = np.shape(center)[0]
    print('样本集样本总数：\n', count)
    # 2.2 求样本矩阵的协方差矩阵
    M = np.dot(center.T, center)/(count - 1)
    print('样本的协方差矩阵M：\n', M)
    return M

# 4、从特征值由大到小排序，取前K个组成 降维转换矩阵
def getReduction(values, vectors):
    # 4.1 将特征值从大到小排序
    orded = np.argsort(-1 * values)
    # 4.2 取前 K 个特征向量组成 降维转换矩阵（每一列代表一个元素）
    beforeReduction = [vectors[:,orded[i]] for i in range(K)]
    # 4.3 将列转换为行，即转置
    reduction = np.transpose(beforeReduction)
    print('降维转置矩阵：\n', reduction)
    return reduction


# 3、求协方差矩阵的特征值和特征向量
def getEigenValuesAndVector(cova):
    # 3.1 求协方差矩阵的特征值、特征向量
    values, vectors = np.linalg.eig(cova)
    print('样本集协方差矩阵的特征值:\n', values)
    print('样本集协方差矩阵的特征向量:\n', vectors)
    # 4、求特征值由大到小排序，取前K个，求降维转换矩阵
    R = getReduction(values, vectors)
    return R

# 5、通过 原矩阵和降维转换矩阵相乘，得到 降维后的矩阵
def getReductionResult(src, R):
    result = np.dot(src, R)
    print('求得降维后矩阵为：\n', result)
    return result

if __name__ == '__main__':
    # 样本集为 10个样本 3个特征
    src = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    print('样本集为10x3矩阵：\n', src)
    center = centralization(src)
    cova = getCova(center)
    reduction = getEigenValuesAndVector(cova)
    result = getReductionResult(src, reduction)