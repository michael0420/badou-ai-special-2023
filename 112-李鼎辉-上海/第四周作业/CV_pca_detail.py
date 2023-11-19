import cv2
import numpy as np

def PCA(X, K):
    # 先做中心化
    mean = np.mean(X, axis=0)
    centered = X - mean
    # 求协方差矩阵
    cov_matrix = np.cov(centered, rowvar=False)
    # 求特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    top_k_eigenvectors = eigenvectors[:, :K]
    # 隐射
    reduced_data = np.dot(centered, top_k_eigenvectors)
    return reduced_data

if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
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
    K = 2  # 主成分的数量
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X, K)
    print(pca)
