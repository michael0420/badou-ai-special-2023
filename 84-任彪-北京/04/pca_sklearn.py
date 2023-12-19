import numpy as np
from sklearn.decomposition import PCA

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
    # 降低一维度
    pca = PCA(X.shape[1]-1)
    # 训练
    pca.fit(X)
    # 降维后的数据
    newX=pca.fit_transform(X)
    print(pca.explained_variance_ratio_)  #输出贡献率
    print(newX)                  #输出降维后的数据