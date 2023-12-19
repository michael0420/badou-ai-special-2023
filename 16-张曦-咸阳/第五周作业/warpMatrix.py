import numpy as np


def WarpMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((nums * 2, 8))
    B = np.zeros((nums * 2, 1))  # A*warpMatrix=B
    for i in range(0, nums):
        A_i = src[i, :]  # [x , y]
        B_i = dst[i, :]  # [x', y']

        A[i * 2, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[i * 2] = B_i[0]

        A[i * 2 + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[i * 2 + 1] = B_i[1]
    A = np.mat(A)
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


print('warpMatrix')
src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
src = np.array(src)

dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
dst = np.array(dst)

warpMatrix = WarpMatrix(src, dst)
print(warpMatrix)
