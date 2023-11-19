
import numpy as np


def centralized(X):

    # 矩阵进行转置，因为矩阵表示的行是样本，列是样本的特征 ，转置以后，获取所有样本在一个特征上的均值，
    meanList = []
    for attr in X.T:
        avg = np.mean(attr)
        meanList.append(avg)
    means = np.array(meanList)
    centerx = X - means
    print("新的去中心化后的样本矩阵",centerx)
    return centerx


if __name__ == '__main__':

    x = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])


    # 1、对举证做0均值处理
    center_x = centralized(x)
    print("中心化后的矩阵", center_x)
    # 2、求协方差矩阵 根据公式得协方差矩阵D = (Z.T * Z) / m (代表样本个数)
    sample_num = center_x.shape[0]
    print("样本数", sample_num)
    # np.dot 代表矩阵乘，* 代表对应位置乘
    cov = np.dot(center_x.T , center_x) / sample_num
    print("协方差矩阵", cov)
    # 3、求协方差矩阵的特征值和特征向量  np.linalg.eig 是固定的，列数据代表的是某个特征的向量值
    eigen_vals, eigen_vecs = np.linalg.eig(cov)
    print("特征值",eigen_vals)
    print("特征向量",eigen_vecs)
    #4、确定降维后的维度个数
    eigen_length = x.shape[1] - 1
    print("要获取的特征",eigen_length)
    # 5、排序获取前top eigen_length 的维度
    eigen_length = x.shape[1]-1
    print("获取前top n 的向量",eigen_length)
    # 对特征值排完序，将对应索引传递给特征向量的列，就可以得到对应排序的特征向量。
    indexList = np.argsort(-eigen_vals)
    new_eigen_vecs = eigen_vecs[:,indexList[:eigen_length]]
    print("截取后的，",new_eigen_vecs)
    #4、原矩阵映射到特征矩阵上
    result = np.dot(x,new_eigen_vecs)
    print("最后的降维特征矩阵",result)

