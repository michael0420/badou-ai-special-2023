import cv2
import numpy as np


def generate_warp_matrix(src, dst):
    # src 与 dst 必须具有相同的行数 且不能小于4
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    # nums 对应点的数量
    nums = src.shape[0]
    # A * warp_matrix = B
    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))
    for i in range(nums):
        # A_i 第i个源图像上的点坐标
        A_i = src[i, :]
        # B_i 第i个目标图像上的点坐标
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i, :] = B_i[0]
        B[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    # 矩阵类型转换
    A = np.mat(A)
    warp_matrix = A.I * B

    # 将结果处理为3 * 3 的透视变换矩阵
    warp_matrix = np.array(warp_matrix).T[0]
    # a33 = 1
    warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0, axis=0)
    warp_matrix = warp_matrix.reshape((3, 3))

    return warp_matrix


if __name__ == '__main__':
    img = cv2.imread("../Images/ipad_test.jpg")
    src = np.float32([[216, 581], [778, 351], [566, 1397], [1116, 1139]])
    dst = np.float32([[0, 0], [608, 0], [0, 890], [608, 890]])
    # warp_matrix = generate_warp_matrix(src, dst)
    warp_matrix = cv2.getPerspectiveTransform(src, dst)
    res = cv2.warpPerspective(img.copy(), warp_matrix, (608, 890))
    cv2.imshow("src", img)
    cv2.imshow("res", res)
    cv2.waitKey(0)
