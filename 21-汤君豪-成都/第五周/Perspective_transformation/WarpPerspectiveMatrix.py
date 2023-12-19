import numpy as np

def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    nums = src.shape[0]
    # A*WarpPerspectiveMatrix=B
    A = np.zeros((2*nums, 8))
    B = np.zeros((2*nums, 1))
    for i in range(2*nums):
        # 在奇数行
        if (i+1)%2 != 0:
            A[i, 0:2] = src[i // 2]
            A[i, 2] = 1
            A[i, -2] = -src[i//2][0] * dst[i//2][0]
            A[i, -1] = -src[i//2][1] * dst[i//2][0]
            B[i] = dst[i // 2][0]
        # 在偶数行
        else:
            A[i, 3:5] = src[i // 2]
            A[i, 5] = 1
            A[i, -2] = -src[i // 2][0] * dst[i // 2][1]
            A[i, -1] = -src[i // 2][1] * dst[i // 2][1]
            B[i] = dst[i // 2][1]
    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    matrix = A.I * B
    # 之后为结果的后处理
    matrix = np.array(matrix).T[0]
    # 插入a_33 = 1
    matrix = np.insert(matrix, matrix.shape[0], values=1.0, axis=0)
    matrix = matrix.reshape((3, 3))
    return matrix

if __name__ == "__main__":
    print("WarpPerspectiveMatrix")
    src = np.array([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])
    dst = np.array([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])
    matrix = WarpPerspectiveMatrix(src, dst)
    print(matrix)