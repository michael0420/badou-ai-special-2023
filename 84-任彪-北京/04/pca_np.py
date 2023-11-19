import numpy as np


def fit_transform(X,K):
        # 求协方差矩阵
        X = X - X.mean(axis=0)
        covariance = np.dot(X.T, X) / X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        components = eig_vectors[:, idx[:K]]
        # 对X进行降维
        return np.dot(X, components)

if __name__ == '__main__':
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    # 调用
    newX = fit_transform(X,X.shape[1]-1)
    print("降维后的矩阵数据",newX)  # 输出降维后的数据
