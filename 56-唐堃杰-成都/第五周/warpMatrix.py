# 计算透视变换矩阵
import cv2.xfeatures2d
import numpy as np


def warpPerspectiveMatrix(src, dst):
    # 断言，确保原图与目标图片具有相同的行数，并且至少有4行
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    lineNums = src.shape[0]
    A = np.zeros(2 * lineNums, 8)
    B = np.zeros(2 * lineNums, 1)

    for i in range(0, lineNums):
        # 获取源点和目标点的第i行
        A_i = src[i:]
        B_i = dst[i:]
        # 这里满足X'这第一个公式
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        # 目标点x坐标的结果
        B[2 * i] = B_i[0]
        # 满足Y'这个公式
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        # 目标点y坐标的结果
        B[2 * i + 1] = B_i[1]
    # 将A数组转化为矩阵对象
    A = np.mat(A)
    # 求解A⋅warpMatrix = B
    warpMatrix = A.I * B
    # 将其转换为矩阵并要第一行的解
    warpMatrix = np.array(warpMatrix).T[0]
    # 在末尾插入设置为1的a33
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    # 这里重塑矩阵为3 x 3的矩阵
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    warpMatrix = warpPerspectiveMatrix(src, dst)
    print(warpMatrix)
