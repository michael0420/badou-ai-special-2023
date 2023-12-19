import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    # 定义图片文件路径
    pic_path = 'lenna.png'
    # 使用Matplotlib库的imread函数读取指定路径的图片文件，存储在变量img中
    img = plt.imread(pic_path)
    # 检查文件路径的后缀是否为'.png'
    if pic_path[-4:] == '.png':
        # 如果图片格式为'.png'，将像素值从0到1的范围扩展到0到255的范围
        img = img * 255
    # 将图像转换为灰度图像，计算每个像素点在RGB通道上的平均值
    img = img.mean(axis=-1)

    # 1、高斯平滑
    # 高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5
    # 计算高斯核的维度，使用四舍五入函数
    dim = int(np.round(6 * sigma + 1))
    # 如果维度是偶数，则加一，确保高斯核的维度是奇数
    if dim % 2 == 0:
        dim += 1
    # 创建一个二维数组，用于存储高斯核的值
    Gaussian_filter = np.zeros([dim, dim])
    # 生成一个序列，用于后续计算高斯核的值
    tmp = [i - dim // 2 for i in range(dim)]
    # 定义一个变量n1，它是高斯函数公式的一部分，其中sigma是标准差，math.pi是π的值
    n1 = 1 / (2 * math.pi * sigma ** 2)
    # 定义一个变量n2，它也是高斯函数公式的一部分，用于计算指数部分
    n2 = -1 / (2 * sigma ** 2)
    # 开始一个双重循环，遍历高斯核的每一个元素
    for i in range(dim):
        for j in range(dim):
            # 计算高斯核的每一个元素值，使用高斯函数的公式
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    # 将计算得到的高斯核归一化，使得高斯核的所有元素之和为1
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    # 获取输入图像的尺寸
    dx, dy = img.shape
    # 创建一个与输入图像相同大小的全零数组，用于存储平滑后的图像
    img_new = np.zeros(img.shape)
    # 边缘填补输入图像，确保后续处理不会超出边界
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
    # 对输入图像进行高斯平滑操作
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
    # 使用Matplotlib库显示平滑后的图像
    plt.figure(1)
    # 将浮点型数据转换为整数类型，显示为灰度图像
    plt.imshow(img_new.astype(np.uint8), "gray")
    # 关闭坐标轴的显示
    plt.axis('off')

    # 2、求梯度。以下两个是滤波求梯度用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    # 定义Sobel算子的水平和垂直卷积核
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # 初始化用于存储梯度图像和进行边缘填补的变量
    # 存储梯度图像的x方向分量
    img_tidu_x = np.zeros(img_new.shape)
    # 存储梯度图像的y方向分量
    img_tidu_y = np.zeros([dx, dy])
    # 存储合并后的梯度图像
    img_tidu = np.zeros(img_new.shape)
    # 边缘填补，根据上面矩阵结构所以写1
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
    # 计算梯度图像的x和y方向分量，并合并为梯度幅值
    for i in range(dx):
        for j in range(dy):
            # 计算梯度图像的x方向分量
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            # 计算梯度图像的y方向分量
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            # 计算梯度幅值，合并x和y方向的分量
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
            # 处理可能出现的除零情况，避免在计算梯度方向时出现错误
    # 添加一个小的正数，避免除以零
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    # 计算梯度方向
    angle = img_tidu_y / img_tidu_x
    # 使用Matplotlib库将梯度图像显示为灰度图像，并关闭坐标轴的显示
    plt.figure(2)
    # 将梯度图像转换为整数类型，显示为灰度图像
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    # 关闭坐标轴的显示
    plt.axis('off')

    # 3、非极大值抑制
    # 创建一个与输入图像形状相同的全零矩阵，用来存储处理后的图像
    img_yizhi = np.zeros(img_tidu.shape)
    # 遍历输入图像的像素，但排除了边界像素
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            # 初始化一个标志变量，用于标记在8邻域内是否需要保留当前像素
            flag = True
            # 创建一个3x3的矩阵，表示当前像素的8邻域梯度幅值矩阵
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]
            # 根据角度的不同，使用不同的插值方式，比较像素的梯度幅值与相邻像素的插值值
            # 如果满足条件，将flag设置为True，否则为False
            # 如果当前像素的梯度方向小于等于-1，表示梯度近似垂直于图像边缘
            if angle[i, j] <= -1:
                # 计算两个相邻像素的线性插值值，用于边缘抑制
                # 计算num_1的值，通过邻域内右侧像素和中心像素的梯度变化率与梯度方向的比值，再加上右侧像素的梯度幅值
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                # 计算num_2的值，通过邻域内左侧像素和中心像素的梯度变化率与梯度方向的比值，再加上左侧像素的梯度幅值
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                # 如果当前像素的梯度幅值不同时大于插值值，将标志变量设置为False
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            # 如果当前像素的梯度方向大于等于1，表示梯度近似水平于图像边缘
            elif angle[i, j] >= 1:
                # 计算两个相邻像素的线性插值值，用于边缘抑制
                # 计算num_1的值，通过邻域内右侧两个像素的梯度变化率与梯度方向的比值，再加上右侧中心像素的梯度幅值
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                # 计算num_2的值，通过邻域内左侧两个像素的梯度变化率与梯度方向的比值，再加上左侧中心像素的梯度幅值
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                # 如果当前像素的梯度幅值不同时大于插值值，将标志变量设置为False
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            # 如果当前像素的梯度方向介于0和1之间，表示梯度近似为水平和垂直的组合
            elif angle[i, j] > 0:
                # 计算两个相邻像素的线性插值值，用于边缘抑制
                # 计算num_1的值，通过邻域内上侧两个像素的梯度变化率与梯度方向的乘积，再加上上侧中心像素的梯度幅值
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                # 计算num_2的值，通过邻域内下侧两个像素的梯度变化率与梯度方向的乘积，再加上下侧中心像素的梯度幅值
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                # 如果当前像素的梯度幅值不同时大于插值值，将标志变量设置为False
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            # 如果当前像素的梯度方向介于-1和0之间，表示梯度介于垂直和水平之间
            elif angle[i, j] < 0:
                # 计算两个相邻像素的线性插值值，用于边缘抑制
                # 计算num_1的值，通过邻域内左侧两个像素的梯度变化率与梯度方向的乘积，再加上左侧中心像素的梯度幅值
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                # 计算num_2的值，通过邻域内右侧两个像素的梯度变化率与梯度方向的乘积，再加上右侧中心像素的梯度幅值
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                # 如果当前像素的梯度幅值不同时大于插值值，将标志变量设置为False
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            # 如果flag为True，将当前像素的梯度幅值赋值给img_yizhi矩阵的相应位置
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    # 将处理后的图像显示出来，img_yizhi矩阵被转换为无符号整数类型，并以灰度图像的形式显示在第三个图像窗口中，同时关闭坐标轴
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    # 计算图像像素的均值并计算低阈值
    lower_boundary = img_tidu.mean() * 0.5
    # 根据低阈值计算高阈值，高阈值是低阈值的三倍
    high_boundary = lower_boundary * 3
    # 存储边缘像素的坐标的空列表
    zhan = []
    # 遍历图像的像素，忽略外圈像素
    for i in range(1, img_yizhi.shape[0] - 1):
        for j in range(1, img_yizhi.shape[1] - 1):
            # 如果像素值大于等于高阈值，标记为边缘像素，将坐标存储到zhan列表
            if img_yizhi[i, j] >= high_boundary:
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            # 如果像素值小于等于低阈值，标记为非边缘像素
            elif img_yizhi[i, j] <= lower_boundary:
                img_yizhi[i, j] = 0
    # 使用深度优先搜索算法进行边缘像素扩展
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        # 获取当前像素周围3x3区域的像素值
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    # 检查周围的8个像素，如果符合条件，标记为边缘像素并加入栈中
    # 这里是一个循环，检查3x3区域内的所有像素
    # 如果像素值在高阈值和低阈值之间，将其标记为边缘像素，并加入栈中
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        # 将左上角的像素标记为边缘
        img_yizhi[temp_1 - 1, temp_2 - 1] = 255
        # 将左上角的坐标加入栈
        zhan.append([temp_1 - 1, temp_2 - 1])
        # 如果像素值在高阈值和低阈值之间，将其标记为边缘像素，并加入栈中
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        # 将上方的像素标记为边缘
        img_yizhi[temp_1 - 1, temp_2] = 255
        # 将上方的坐标加入栈
        zhan.append([temp_1 - 1, temp_2])
        # 如果像素值在高阈值和低阈值之间，将其标记为边缘像素，并加入栈中
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        # 将右上角的像素标记为边缘
        img_yizhi[temp_1 - 1, temp_2 + 1] = 255
        # 将右上角的坐标加入栈
        zhan.append([temp_1 - 1, temp_2 + 1])
        # 如果像素值在高阈值和低阈值之间，将其标记为边缘像素，并加入栈中
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        # 将左边的像素标记为边缘
        img_yizhi[temp_1, temp_2 - 1] = 255
        # 将左边的坐标加入栈
        zhan.append([temp_1, temp_2 - 1])
        # 如果像素值在高阈值和低阈值之间，将其标记为边缘像素，并加入栈中
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        # 将右边的像素标记为边缘
        img_yizhi[temp_1, temp_2 + 1] = 255
        # 将右边的坐标加入栈
        zhan.append([temp_1, temp_2 + 1])
        # 如果像素值在高阈值和低阈值之间，将其标记为边缘像素，并加入栈中
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        # 将左下角的像素标记为边缘
        img_yizhi[temp_1 + 1, temp_2 - 1] = 255
        # 将左下角的坐标加入栈
        zhan.append([temp_1 + 1, temp_2 - 1])
        # 如果像素值在高阈值和低阈值之间，将其标记为边缘像素，并加入栈中
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        # 将下方的像素标记为边缘
        img_yizhi[temp_1 + 1, temp_2] = 255
        # 将下方的坐标加入栈
        zhan.append([temp_1 + 1, temp_2])
        # 如果像素值在高阈值和低阈值之间，将其标记为边缘像素，并加入栈中
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        # 将右下角的像素标记为边缘
        img_yizhi[temp_1 + 1, temp_2 + 1] = 255
        # 将右下角的坐标加入栈
        zhan.append([temp_1 + 1, temp_2 + 1])
        # 将非边缘像素标记为0
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
    # 使用Matplotlib库绘制处理后的图像
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    # 关闭坐标刻度值
    plt.axis('off')
    plt.show()
