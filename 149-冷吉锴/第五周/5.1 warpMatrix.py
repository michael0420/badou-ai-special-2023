import numpy as np


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4  # 防呆检查，输入和输出图的shape相等

    nums = src.shape[0]  # 4对点
    A = np.zeros((2 * nums, 8))  # A * warpMatrix = B [8,8]
    B = np.zeros((2 * nums, 1))  # [8,1]
    # 代入公式
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        # 求矩阵A
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        # 求矩阵B
        B[2 * i] = B_i[0]
        # 求矩阵A
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        # 求矩阵B
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)  # 将数组转换成矩阵的函数
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == '__main__':
    print('warpMatrix')
    src = np.array([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])
    dst = np.array([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])

    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
