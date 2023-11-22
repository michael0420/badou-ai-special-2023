import numpy as np
 
def WarpPerspectiveMatrix(src, dst):
    # 检查src和dst的形状是否相同且大于等于4。如果不满足条件，程序将抛出异常。
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    # 获取src的行数，并将其赋值给变量nums。
    nums = src.shape[0]

    # 创建一个形状为(2*nums, 8)的全零矩阵，并将其赋值给变量A。
    A = np.zeros((2*nums, 8))  # A*warpMatrix=B

    # 创建一个形状为(2*nums, 1)的全零矩阵，并将其赋值给变量B。
    B = np.zeros((2*nums, 1))

    # 使用循环遍历src的每一行。
    for i in range(0, nums):

        # 获取src的第i行，并将其赋值给变量A_i。
        # 获取dst的第i行，并将其赋值给变量B_i。
        A_i = src[i, :]
        B_i = dst[i, :]

        # 将A_i的前三个元素、后四个元素（分别为0、0、0）以及-A_i[0]*B_i[0]和-A_i[1]*B_i[0]组成一个列表，并将其赋值给A的第2*i行。
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]

        # 将B_i的第一个元素赋值给B的第2*i行。
        B[2*i] = B_i[0]

        # 将A_i的前三个元素、后四个元素（分别为0、0、0）以及-A_i[0]*B_i[1]和-A_i[1]*B_i[1]组成一个列表，并将其赋值给A的第2*i+1行。
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]

        # 将B_i的第二个元素赋值给B的第2*i+1行。
        B[2*i+1] = B_i[1]

    # 将矩阵A转换为NumPy矩阵对象。
    A = np.mat(A)
    #  用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    
    # 之后为结果的后处理
    # 将矩阵warpMatrix转换为NumPy数组，并对其进行转置操作，然后取第一行作为新的矩阵warpMatrix。
    warpMatrix = np.array(warpMatrix).T[0]

    # 在矩阵warpMatrix的最后一行插入值为1.0的元素。
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1

    # 将矩阵warpMatrix重新调整为形状为(3, 3)的矩阵。
    warpMatrix = warpMatrix.reshape((3, 3))

    # 返回计算得到的透视变换矩阵。
    return warpMatrix
 
if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)
    
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
