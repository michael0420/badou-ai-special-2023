import numpy as np
import matplotlib.pyplot as plt
import math


# 接口
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("canny", cv2.Canny(gray, 200, 300))
cv2.waitKey()
cv2.destroyAllWindows()



if __name__ == '__main__':
    img_path = 'lenna.png'
    img = plt.imread(img_path)
    if img_path[-4:] == '.png':
        img *= 255
    img_gray = img.mean(axis=-1)
    # height, width = img.shape[:2]
    # img_gray = np.zeros((height, width))
    # # 灰度化
    # for h in range(height):
    #     for w in range(width):
    #         cur = img[h, w]
    #         img_gray[h, w] = int(cur[0] * 0.11 + cur[1] * 0.59 + cur[2] * 0.3)
    # plt.imshow(img_gray, cmap='gray')
    # print(img_gray)
    # plt.show()
    # print(img.shape, img)

    # 高斯滤波
    sigma = 0.5
    dim = int(np.round(sigma * 6 + 1))
    if dim % 2 == 0:
        dim += 1
    Gauss_filter = np.zeros([dim, dim])
    tmp = [i - dim // 2 for i in range(dim)]
    # print(tmp)
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gauss_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gauss_filter = Gauss_filter / Gauss_filter.sum()
    dx, dy = img_gray.shape
    img_new_gray = np.zeros(img_gray.shape)
    pad_num = dim // 2
    img_pad = np.pad(img_gray, ((pad_num, pad_num), (pad_num, pad_num)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_new_gray[i, j] = np.sum(img_pad[i: i + dim, j: j + dim] * Gauss_filter)
    plt.figure(1)
    plt.imshow(img_new_gray.astype(np.uint8), cmap='gray')
    plt.axis('off')
    # print(img_new_gray)
    # 梯度 算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_grade_x = np.zeros(img_new_gray.shape)
    img_grade_y = np.zeros(img_new_gray.shape)
    img_grade = np.zeros(img_new_gray.shape)
    img_sobel_pad = np.pad(img_new_gray, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_grade_x[i, j] = np.sum(sobel_x * img_sobel_pad[i: i + 3, j: j + 3])
            img_grade_y[i, j] = np.sum(sobel_y * img_sobel_pad[i: i + 3, j: j + 3])
            img_grade[i, j] = np.sqrt(img_grade_x[i, j] ** 2 + img_grade_y[i, j] ** 2)
    # print(img_grade_x == 0)
    img_grade_x[img_grade_x == 0] = 0.00000001
    # print(img_grade_x)
    tan_value = img_grade_y / img_grade_x
    plt.figure(2)
    plt.imshow(img_grade.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # print(img_grade)
    # print(tan_value)
    # print(img_grade_x)
    # print(img_grade_y)

    # 非极大值抑制
    img_suppression = np.zeros(img_new_gray.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            tmp = img_grade[i - 1: i + 2, j - 1: j + 2]
            flag = True  # 是否需要抑制 True: 保留原值即不抑制
            if tan_value[i, j] <= -1:
                # abs(y) >= abs(x)时, θ的余角对应的tan值即某点距亚像素点的距离
                # 即tan(90° - θ) = 1 / tanθ，tanθ <= -1时，权重weight = -1 / tanθ,
                # 反之tanθ >= 1 时weight = 1 / tanθ, 故有:
                # -1 / tan_value * tmp[0, 0] + (1 + 1 / tan_value) * tmp[0, 1]
                # 或 1 / tan_value * tmp[0, 0] + (1 - 1 / tan_value) * tmp[0, 1]
                num_1 = (tmp[0, 1] - tmp[0, 0]) / tan_value[i, j] + tmp[0, 1]
                num_2 = (tmp[2, 1] - tmp[2, 2]) / tan_value[i, j] + tmp[2, 1]
                if not (img_grade[i, j] > num_1 and img_grade[i, j] > num_2):
                    flag = False
            elif tan_value[i, j] < 0:
                # abs(y) < abs(x)时, tanθ即某点距亚像素点距离, tanθ <= 0时, 权重
                # weight = tanθ 反之weight = -tanθ 故有:
                # -tan_value * tmp[0, 0] + (1 + tan_value) * tmp[0, 1]
                # 或 tan_value * tmp[0, 0] + (1 - tan_value) * tmp[0, 1]
                num_1 = (tmp[1, 0] - tmp[0, 0]) * tan_value[i, j] + tmp[1, 0]
                num_2 = (tmp[1, 2] - tmp[2, 2]) * tan_value[i, j] + tmp[1, 2]
                if not (img_grade[i, j] > num_1 and img_grade[i, j] > num_2):
                    flag = False
            elif tan_value[i, j] >= 1:
                num_1 = (tmp[0, 2] - tmp[0, 1]) / tan_value[i, j] + tmp[0, 1]
                num_2 = (tmp[2, 0] - tmp[2, 1]) / tan_value[i, j] + tmp[2, 1]
                if not (img_grade[i, j] > num_1 and img_grade[i, j] > num_2):
                    flag = False
            elif tan_value[i, j] > 0:
                num_1 = (tmp[0, 2] - tmp[1, 2]) * tan_value[i, j] + tmp[1, 2]
                num_2 = (tmp[2, 0] - tmp[1, 0]) * tan_value[i, j] + tmp[1, 0]
                if not (img_grade[i, j] > num_1 and img_grade[i, j] > num_2):
                    flag = False
            if flag:
                img_suppression[i, j] = img_grade[i, j]
    plt.figure(3)
    plt.imshow(img_suppression.astype(np.uint8), cmap='gray')
    plt.axis('off')
    # 双阈值检测
    lower_boundary = img_grade.mean() * 0.5
    high_boundary = lower_boundary * 3
    stack = []

    # print(lower_boundary, high_boundary)

    for i in range(1, img_suppression.shape[0] - 1):
        for j in range(1, img_suppression.shape[1] - 1):
            if img_suppression[i, j] >= high_boundary:
                img_suppression[i, j] = 255
                stack.append([i, j])
            elif img_suppression[i, j] <= lower_boundary:
                img_suppression[i, j] = 0
    # print(len(stack), stack)

    while not len(stack) == 0:
        i, j = stack.pop()
        tmp = img_suppression[i-1: i+2, j-1: j+2]
        if lower_boundary < tmp[0, 0] < high_boundary:
            img_suppression[i - 1, j - 1] = 255
            stack.append([i - 1, j - 1])
        if lower_boundary < tmp[0, 1] < high_boundary:
            img_suppression[i - 1, j] = 255
            stack.append([i - 1, j])
        if lower_boundary < tmp[0, 2] < high_boundary:
            img_suppression[i - 1, j + 1] = 255
            stack.append([i - 1, j + 1])
        if lower_boundary < tmp[1, 0] < high_boundary:
            img_suppression[i, j - 1] = 255
            stack.append([i, j - 1])
        if lower_boundary < tmp[1, 2] < high_boundary:
            img_suppression[i, j + 1] = 255
            stack.append([i, j + 1])
        if lower_boundary < tmp[2, 0] < high_boundary:
            img_suppression[i + 1, j - 1] = 255
            stack.append([i + 1, j - 1])
        if lower_boundary < tmp[2, 1] < high_boundary:
            img_suppression[i + 1, j] = 255
            stack.append([i + 1, j])
        if lower_boundary < tmp[0, 1] < high_boundary:
            img_suppression[i + 1, j + 1] = 255
            stack.append([i + 1, j + 1])
    # for h in range(i-1, i+2):
        #     for w in range(j-2, j+2):
        #         tmp = img_suppression[h, w]
        #         if high_boundary > tmp > lower_boundary:
        #             img_suppression[h, w] = 255
        #             stack.append([h, w])
    for i in range(img_suppression.shape[0]):
        for j in range(img_suppression.shape[1]):
            if img_suppression[i, j] != 0 and img_suppression[i, j] != 255:
                img_suppression[i, j] = 0
    # print(img_suppression)
    plt.figure(4)
    plt.imshow(img_suppression.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()




