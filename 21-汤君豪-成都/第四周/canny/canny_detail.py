# 1.图像灰度化
import cv2
# (512, 512, 3)
# img = cv2.imread('lenna.png')
# img_G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
import matplotlib.pyplot as plt
pic_path = 'lenna.png'
img = plt.imread(pic_path)
if pic_path[-4:] == '.png':    # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
    img = img * 255            # 还是浮点数类型
img_G = img.mean(axis=-1)      # 取均值就是灰度化了

# 2.图像高斯滤波平滑
# 高斯平滑时的高斯核参数，标准差，可调
sigma = 0.5
# round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
import numpy as np
dim = int(np.round(6*sigma + 1))
# 最好是奇数,不是的话加一
if dim % 2 == 0:
    dim += 1
# 创建高斯核，这是数组不是列表
Gaussian_filter = np.zeros([dim, dim])
# 生成一个序列，[-2, -1, 0, 1, 2]
tmp = [i-dim//2 for i in range(dim)]
# 计算高斯核
import math
n1 = 1/(2*math.pi*sigma**2)
n2 = -1/(2*sigma**2)
for i in range(dim):
    for j in range(dim):
        Gaussian_filter[i][j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
# (512, 512)
dx, dy = img_G.shape
# 存储平滑之后的图像，zeros函数得到的是浮点型数据
img_new = np.zeros([dx, dy])
# 原图边缘填补
tmp = dim//2
img_G_pad = np.pad(img_G, ((tmp, tmp), (tmp, tmp)), 'constant')
for i in range(dx):
    for j in range(dy):
        img_new[i, j] = np.sum(img_G_pad[i:i+dim, j:j+dim]*Gaussian_filter)
plt.figure(1)
# 此时的img_new是255的浮点型数据，强制类型转换成np.uint8才可以，gray灰阶
plt.imshow(img_new.astype(np.uint8), cmap='gray')
# plt.imshow(img_new.astype(np.uint8))
# 关闭坐标刻度值
plt.axis('off')

# 3.求梯度。检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
# 存梯度值
img_Gradient_x = np.zeros(img_new.shape)
img_Gradient_y = np.zeros(img_new.shape)
img_Gradient = np.zeros(img_new.shape)
# 边缘填补，根据上面矩阵结构所以写1
img_new_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
for i in range(dx):
    for j in range(dy):
        # x方向
        img_Gradient_x[i][j] = np.sum(img_new_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
        # y方向
        img_Gradient_y[i][j] = np.sum(img_new_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
        # 对角
        img_Gradient[i][j] = np.sqrt(img_Gradient_x[i][j] ** 2 + img_Gradient_y[i][j] ** 2)
# 防止后面除以img_tidu_x时，除数为0
img_Gradient_x[img_Gradient_x == 0] = 0.00000001
# angle = np.arctan(img_Gradient_y/img_Gradient_x)
angle = img_Gradient_y/img_Gradient_x
# for i in range(angle.shape[0]):
#     for j in range(angle.shape[1]):
#         print(angle[i][j])
plt.figure(2)
plt.imshow(img_Gradient.astype(np.uint8), cmap='gray')
plt.axis('off')

# 4.非极大值抑制
img_nonmaximum = np.zeros(img_Gradient.shape)
'''
# 这是一段失败的代码
# 设置图像的边缘为不可能的边界点
for i in range(1, img_Gradient.shape[0]-1):
    for j in range(1, img_Gradient.shape[1]-1):
        flag = True
        temp = img_Gradient[i-1:i+2, j-1:j+2]
        # 当前点的梯度，x方向的，y方向的
        current_g = img_Gradient[i][j]
        current_gx = img_Gradient_x[i][j]
        current_gy = img_Gradient_y[i][j]
        # 如果方向导数y分量比x分量大，说明导数方向趋向于y分量
        if(math.fabs(current_gy) > math.fabs(current_gx)):
            # 计算权重
            weight = math.fabs(current_gx) / math.fabs(current_gy)
            g2 = img_Gradient[i - 1][j]
            g4 = img_Gradient[i + 1][j]
            # 如果gx与gy同向，即同号
            if (current_gx * current_gy > 0):
                g1 = img_Gradient[i - 1][j - 1]
                g3 = img_Gradient[i + 1][j + 1]
            # 如果gx与gy反向，即异号
            else:
                g1 = img_Gradient[i - 1][j + 1]
                g3 = img_Gradient[i + 1][j - 1]
        # 如果方向导数x分量比y分量大，说明导数方向趋向于x分量
        else:
            # 计算权重
            weight = math.fabs(current_gy) / math.fabs(current_gx)
            g2 = img_Gradient[i][j - 1]
            g4 = img_Gradient[i][j + 1]
            # 如果gx与gy同向，即同号
            if (current_gx * current_gy > 0):
                g1 = img_Gradient[i - 1][j - 1]
                g3 = img_Gradient[i + 1][j + 1]
            # 如果gx与gy反向，即异号
            else:
                g1 = img_Gradient[i + 1][j - 1]
                g3 = img_Gradient[i - 1][j + 1]
        gA = (1/weight) * g1 + (1 - weight) * g2
        gB = weight * g3 + (1 - weight) * g4
        if(current_g > gA and current_g > gB):
            img_nonmaximum[i][j] = current_g
'''
for i in range(1, img_Gradient.shape[0] - 1):
    for j in range(1, img_Gradient.shape[1] - 1):
        # 与当前梯度值的8邻域内比较是否要抹去
        flag = False
        # 取当前梯度值的8领域梯度
        temp = img_Gradient[i - 1:i + 2, j - 1:j + 2]
        angle_current = angle[i][j]
        img_Gradient_current = img_Gradient[i, j]
        # tan45 < tan@ < tan90
        if angle_current >= 1:
            g1 = temp[0, 1]
            g2 = temp[0, 2]
            g3 = temp[2, 1]
            g4 = temp[2, 0]
            num_1 = g2 / angle_current + (1 - 1 / angle_current) * g1
            num_2 = g4 / angle_current + (1 - 1 / angle_current) * g3
            if img_Gradient_current > num_1 and img_Gradient_current > num_2:
                flag = True
        # tan0 < tan@ < tan45
        elif angle_current < 1 and angle_current >= 0:
            g1 = temp[1, 2]
            g2 = temp[0, 2]
            g3 = temp[1, 0]
            g4 = temp[2, 0]
            num_1 = g2 * angle_current + (1 - angle_current) * g1
            num_2 = g4 * angle_current + (1 - angle_current) * g3
            if img_Gradient_current > num_1 and img_Gradient_current > num_2:
                flag = True
        # tan90 < tan@ < tan135
        elif angle_current <= -1:
            g1 = temp[0, 1]
            g2 = temp[0, 0]
            g3 = temp[2, 1]
            g4 = temp[2, 2]
            num_1 = g2 * (-1 / angle_current) + (1 + 1 / angle_current) * g1
            num_2 = g4 * (-1 / angle_current) + (1 + 1 / angle_current) * g3
            if img_Gradient_current > num_1 and img_Gradient_current > num_2:
                flag = True
        # tan135 < tan@ < tan180
        elif angle_current < 0 and angle_current > -1:
            g1 = temp[1, 0]
            g2 = temp[0, 0]
            g3 = temp[1, 2]
            g4 = temp[2, 2]
            num_1 = g2 * (-angle_current) + (1 + angle_current) * g1
            num_2 = g4 * (-angle_current) + (1 + angle_current) * g3
            if img_Gradient_current > num_1 and img_Gradient_current > num_2:
                flag = True
        if flag:
            img_nonmaximum[i, j] = img_Gradient[i, j]
plt.figure(3)
plt.imshow(img_nonmaximum.astype(np.uint8), cmap='gray')
plt.axis('off')

# 5.双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
lower_boundary = img_Gradient.mean() * 0.5
higher_boundary = lower_boundary * 3
img_double = img_nonmaximum.copy()
high_pos = []
for i in range(1, img_double.shape[0] - 1):
    for j in range(1, img_double.shape[1] - 1):
        # 大于高阈值，记录为强边缘
        if img_double[i, j] >= higher_boundary:
            img_double[i, j] = 255
            # 将强边缘的位置记录
            high_pos.append([i, j])
        # 小于低阈值，抑制
        elif img_double[i, j] <= lower_boundary:
            img_double[i, j] = 0
while not len(high_pos) == 0:
    a, b = high_pos.pop()
    temp = img_double[a - 1:a + 2, b - 1:b + 2]
    if temp[0, 0] > lower_boundary and temp[0, 0] < higher_boundary:
        img_double[a - 1, b - 1] = 255
        high_pos.append([a - 1, b - 1])
    if temp[0, 1] > lower_boundary and temp[0, 1] < higher_boundary:
        img_double[a - 1, b] = 255
        high_pos.append([a - 1, b])
    if temp[0, 2] > lower_boundary and temp[0, 2] < higher_boundary:
        img_double[a - 1, b + 1] = 255
        high_pos.append([a - 1, b + 1])
    if temp[1, 0] > lower_boundary and temp[1, 0] < higher_boundary:
        img_double[a, b - 1] = 255
        high_pos.append([a, b - 1])
    if temp[1, 2] > lower_boundary and temp[1, 2] < higher_boundary:
        img_double[a, b + 1] = 255
        high_pos.append([a, b + 1])
    if temp[2, 0] > lower_boundary and temp[2, 0] < higher_boundary:
        img_double[a + 1, b - 1] = 255
        high_pos.append([a + 1, b - 1])
    if temp[2, 1] > lower_boundary and temp[2, 1] < higher_boundary:
        img_double[a + 1, b] = 255
        high_pos.append([a + 1, b])
    if temp[2, 2] > lower_boundary and temp[2, 2] < higher_boundary:
        img_double[a + 1, b + 1] = 255
        high_pos.append([a + 1, b + 1])
for i in range(img_double.shape[0]):
    for j in range(img_double.shape[1]):
        if img_double[i, j] != 0 and img_double[i, j] != 255:
            img_double[i, j] = 0
plt.figure(4)
plt.imshow(img_double.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.show()