import numpy as np
import numpy as pn


def PerspectiveTransformation(src, dst):
    if src.shape[0] != dst.shape[0]:
        return -1
    Nums = src.shape[0]
    A = np.zeros((Nums*2, 8))
    B = np.zeros((2*Nums, 1))
    for i in range(src.shape[0]):
        A_i = src[i, :]  #获取数组行数据
        B_i = dst[i, :]
        A[2*i, :] = [A_i[0], A_i[0], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i, :] = B_i[0]
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1, :] = B_i[1]
    A = np.mat(A)
    WarpMatrix = A.I*B
    print(type(WarpMatrix))
    WarpMatrix = np.array(WarpMatrix).T[0]
    WarpMatrix = np.insert(WarpMatrix, WarpMatrix.shape[0], values=1, axis=0)   # 将a33赋值为1
    WarpMatrix = WarpMatrix.reshape((3, 3))
    return WarpMatrix

if __name__=='__main__':
    src = np.array([[25.0, 316.0], [129.0, 32.0], [521.0, 35.0], [613.0, 219.0]])
    dst = np.array([[16.0, 586.0], [51.0, 216.0], [219.0, 61.0], [726.0, 196.0]])
    WarpMatrix = PerspectiveTransformation(src, dst)
    print(WarpMatrix)
