import numpy as np
 
def change(input, output):
    H, W, = input.shape
    H1, W1 = output.shape
    if (H != H1) or (W != W1):
        output = 1
        return output
    
    nums = input.shape[0]
    A = np.zeros((2*nums, 8)) # A*warpMatrix=B
    B = np.zeros((2*nums, 1))
    # 1
    # 根据公式  [X_out]    [a11, a12, a13]  [x_in]
    #          [Y_out] = [a21, a22, a23]  [y_in]
    #          [Z_out]   [a31, a32, a33]  [1]
    # 得到结果[X_out, Y_out, Z_out]
    # X_out = a11*x_in + a12*y_in + a13, Y_out = a21*x_in + a22*y_in + a23, Z_out = a31*x_in + a32*y_in + a33

    # 2
    # Z_out一般化简为1,d得到[X_in/Z_out, Y_in/Z_out, 1]， a33无关且通用所以一般设置为1
    # X_out = a11*x_in + a12*y_in + a13/a31*x_in + a32*y_in + 1
    # Y_out = a21*x_in + a22*y_in + a23/a31*x_in + a32*y_in + 1

    # 3
    # 左右化简X_out和Y_out,得到关系式
    # a11 * x_in + a12 * y_in + a13 - a31 * x_in * X_out - a32 * X_out * y_in = X_out
    # a21 * x_in + a22 * y_in + a23 - a31 * x_in * Y_out - a32 * Y_out * y_in = Y_out
    # 提取系数求a的系列参数,因为有8个未知参数，而每个坐标对只能有两个方程，所以需要4对坐标
    # 每个坐标的比例系数一致，只不过(x_in,y_in),(X_out, Y_out)替换为4个不一样的
    # [x_in, y_in, 1, 0, 0, 0, 0, -x_in * X_out, -y_in * Y_out]    [a11]      [X_out]
    # [0, 0, 0, x_in, y_in, 1, -x_in * Y_out, -y_in * Y_out]       [a12]      [X_out]
    #                                                              [a13]
    #                                                              [a21]
    #                                                              [a22]
    #                                                              [a23]
    #                                                              [a31]
    #                                                              [a32]
    #                                                              [a33] 这个是1,暂时不列式，求前面8个即可


    for i in range(0, nums):
        # 取一对坐标的两个公式
        input_one, output_one = input[i, :], output[i, :]
        A[2*i, :] = [input_one[0], input_one[1], 1, 0, 0, 0, -input_one[0]*output_one[0], -input_one[1]*output_one[0]]
        B[2*i] = output_one[0]

        A[2*i+1, :] = [0, 0, 0, input_one[0], input_one[1], 1, -input_one[0]*output_one[1], -input_one[1]*output_one[1]]
        B[2*i+1] = output_one[1]
 
    A = np.mat(A)
    change_Matrix = A.I * B
    # change_Matrix [a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32]

    # change_Matrix = np.array(change_Matrix).T[0]
    # 需要先加回来a33 = 1,再把向量转化为矩阵
    change_Matrix = np.insert(change_Matrix, change_Matrix.shape[0], values=1.0, axis=0) #插入a_33 = 1
    change_Matrix = change_Matrix.reshape((3, 3))
    return change_Matrix
 
if __name__ == '__main__':
    img_4points = np.array([[207, 151], [517, 285], [17, 601], [343, 731]])
    img_out_4points = np.array([[0, 0], [337, 0], [0, 488], [337, 488]])
    
    change_Matrix = change(img_4points, img_out_4points)
    print(change_Matrix)
