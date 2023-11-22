import cv2
import numpy as np


def wrapMatrix(src, target):
    assert src.shape[0] == target.shape[0] and src.shape[0] >= 4
    # A 初始化一个 8 * 8 的 数组
    A = np.zeros((8, 8))
    # B 初始化一个 8 * 1 的 数组
    B = np.zeros((8, 1))
    # 根据 透视变换里面的列表，设置 A 矩阵和B矩阵，在每个位置上的点
    for index in range(0, 4):
        # 获取原始像素点  数组切片参考：https://blog.csdn.net/qq_41973536/article/details/82690242
        # [i,:] 获取当前维度下的全部数据
        X = src[index,:]
        # 获取透视变化后的像素点
        Y = target[index, :]
        # A 为 8列矩阵 矩阵在 下标为偶数 （0，2，4，6 ） x 上 分别为
        # src[index][0] , src[index][1], 1, 0, 0, -  src[index][0] * target[index][0], - src[index][1] * target[index][0]
        A[2 * index, :] = [X[0], X[1], 1, 0, 0, 0, - X[0] * Y[0], -X[1] * Y[0]]
        # A 矩阵在 下标为基数行 （1，3，5，7） x 上 分别为
        # 0 ，0 ，0 ，src[index][0],src[index][1]， 1, -src[index][0] * target[index][1], - src[index][1] * target[index][1]
        A[2 * index + 1, :] = [0, 0, 0, X[0], X[1], 1, - X[0] * Y[1], -X[1] * Y[1]]
        #B 矩阵在 0，2，4，6 列上 分别为 target[0][0],target[1][0],target[2][0],target[3][0]
        B[2 * index] = Y[0]
        # B 矩阵在 1，3，5，7 列上 分别为 target[0][1],target[1][1],target[2][1],target[3][1]
        B[2 * index + 1] = Y[1]
    # A转换为矩阵
    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    warpMatrix = A.I * B
    #warpMatrix 是一个 8 x 1 的矩阵，需要转成一个 1 x 8 的矩阵，方便插入a33 =1
    warpMatrix = np.array(warpMatrix).T[0]
    # 插入a_33 = 1
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    # 把 数组改成一个 3x3的矩阵
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == '__main__':
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    target = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
    wrapMatrixResult = wrapMatrix(src,target)
    print("透视变换中间矩阵",wrapMatrixResult)
    img = cv2.imread('photo.jpg')
    result = cv2.warpPerspective(img, wrapMatrixResult, (500, 600))
    cv2.imshow("src", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)

