"""
@author: BraHitYQ
Detailed Implementation of Canny Algorithm(canny算法的详细实现)
"""

import numpy as np
import matplotlib.pyplot as plt
import math
 
if __name__ == '__main__':
    pic_path = 'lenna.png' 
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)  # 取均值就是灰度化了
    '''
    Code parsing：
        1.if pic_path[-4:] == '.png':: 这是一个条件判断语句，用于检查图片文件的后缀是否为'.png'。如果后缀是'.png'，则执行以下代码块。
    
        2.img = img * 255: 如果图片文件的后缀是'.png'，则将图片数据乘以255。这一步的目的是将图片的像素值范围从0到1转换为0到255。这是因为PNG格式的图片通常使用8位无符号整数表示像素值，而matplotlib默认使用浮点数表示像素值。通过乘以255，可以将像素值的范围转换为0到255。
    
        3.img = img.mean(axis=-1): 这行代码计算了图片数据的平均值。axis=-1表示沿着最后一个维度（即颜色通道）进行计算。这样可以将彩色图片转换为灰度图像，因为彩色图片通常有三个颜色通道（红、绿、蓝），而灰度图像只有一个通道。通过计算每个像素位置的平均值，可以得到灰度图像的像素值。
    '''
    # ###############################################################################################################1.高斯平滑###########################################################################################################

    # sigma = 1.52  # 高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = int(np.round(6 * sigma + 1))  # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度，根据标准差计算高斯核的大小，这里假设高斯核是一个正方形，边长为dim。
    if dim % 2 == 0:  # 最好是奇数,不是的话加一,判断高斯核的大小是否为偶数，如果是偶数，则加1使其变为奇数。
        dim += 1
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了,创建一个大小为dim x dim的零矩阵,用于存储高斯核的值。
    tmp = [i-dim//2 for i in range(dim)]  # 生成一个序列，表示高斯核在x和y方向上的偏移量。
    n1 = 1/(2*math.pi*sigma**2)  # 计算高斯核，计算高斯函数的归一化系数。
    n2 = -1/(2*sigma**2)  # 计算高斯函数的指数部分。
    for i in range(dim):  # 遍历高斯核的每一行。
        for j in range(dim):  # 遍历高斯核的每一列。
            Gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))  # 计算高斯核的每一个元素的值。
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()  # 将高斯核的所有元素之和归一化为1。
    dx, dy = img.shape  # 获取输入图像的形状，分别表示图像的宽度和高度。
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据，创建一个与输入图像形状相同的零矩阵，用于存储平滑后的图像。
    tmp = dim//2  # 计算边缘填充的大小。
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补，对输入图像进行边缘填充，以便在平滑过程中可以访问到边缘像素。
    for i in range(dx):  # 遍历填充后的图像的每一行。
        for j in range(dy):  # 遍历填充后的图像的每一列。
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)  # 将填充后的图像与高斯核进行卷积操作，得到平滑后的图像的一个像素值。
    plt.figure(1)  # 创建一个新的图形窗口。
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，显示平滑后的图像，使用灰度颜色映射。
    plt.axis('off')  # 关闭坐标轴。

    # ##############################################################################################2.求梯度.以下两个是滤波求梯度用的sobel矩阵(检测图像中的水平、垂直和对角边缘)#######################################################################

    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # 创建一个与输入图像相同大小的零矩阵，用于存储x方向的梯度值
    img_tidu_y = np.zeros([dx, dy])  # 创建一个与输入图像大小相同的零矩阵，用于存储y方向的梯度值
    img_tidu = np.zeros(img_new.shape)  # 创建一个与输入图像相同大小的零矩阵，用于存储总梯度值
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1,# 对输入图像进行边缘填充，以便在计算梯度时能够处理边界像素
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)  # 计算x方向的梯度值
            img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)  # 计算y方向的梯度值
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)  # 计算总梯度值
    img_tidu_x[img_tidu_x == 0] = 0.00000001  # 将x方向梯度值为0的位置替换为一个非常小的值，避免除以零的错误
    angle = img_tidu_y/img_tidu_x  # 计算梯度的方向，即y方向梯度值除以x方向梯度值
    plt.figure(2)  # 创建一个新的图形窗口，编号为2
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')   # 显示总梯度图像，使用灰度颜色映射
    plt.axis('off')

    # ###############################################################################################3.非极大值抑制############################################################################################################################

    img_yizhi = np.zeros(img_tidu.shape)   # 创建一个与输入图像（img_tidu）形状相同的全零矩阵，用于存储处理后的图像（img_yizhi）。
    for i in range(1, dx-1):  # 遍历输入图像的每个像素，注意这里从1开始，到dx-1结束，是为了跳过边界像素。
        for j in range(1, dy-1):  # 同上，遍历输入图像的每个像素，注意这里从1开始，到dy-1结束，是为了跳过边界像素。
            flag = True  # 在8邻域内是否要抹去做个标记,初始化一个标志变量，用于判断当前像素是否满足阈值条件。
            temp = img_tidu[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵,提取以当前像素为中心的3x3邻域。
            if angle[i, j] <= -1:  # 判断当前像素的角度是否小于等于-1(使用线性插值法判断抑制与否)。
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]  # 计算第一个阈值条件中的分子。
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]  # 计算第一个阈值条件中的分母。
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):  # 判断当前像素是否大于两个阈值条件中的较大值。
                    flag = False  # 如果不满足条件，将标志变量设为False。
            elif angle[i, j] >= 1:  # 判断当前像素的角度是否大于等于1。
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]  # 计算第二个阈值条件中的分子。
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]  # 计算第二个阈值条件中的分母。
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):  # 判断当前像素是否大于两个阈值条件中的较大值。
                    flag = False  # 如果不满足条件，将标志变量设为False。
            elif angle[i, j] > 0:  # 判断当前像素的角度是否大于0。
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]  # 计算第三个阈值条件中的分子。
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]  # 计算第三个阈值条件中的分母。
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):  # 判断当前像素是否大于两个阈值条件中的较大值。
                    flag = False  # 如果不满足条件，将标志变量设为False。
            elif angle[i, j] < 0:  # 判断当前像素的角度是否小于0
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]  # 计算第四个阈值条件中的分子。
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]  # 计算第四个阈值条件中的分母。
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):  # 判断当前像素是否大于两个阈值条件中的较大值。
                    flag = False  # 如果不满足条件，将标志变量设为False。
            if flag:  # 如果标志变量为True，表示当前像素满足所有阈值条件。
                img_yizhi[i, j] = img_tidu[i, j]  # 将满足条件的像素值赋给输出图像（img_yizhi）。
    plt.figure(3)  # 创建一个新的图形窗口，编号为3。
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')  # 显示处理后的图像（img_yizhi），使用灰度颜色映射。
    plt.axis('off')

    # ###############################################################################################4.双阈值检测,连接边缘.遍历所有一定是边的点,查看8邻域是否存在有可能是边的点,进栈##################################################################
    # 这一步的目的是确定哪些像素点是真正的边缘，哪些是噪声或者其他非边缘区域。

    lower_boundary = img_tidu.mean() * 0.5  # 计算图像的平均灰度值，并将其乘以0.5作为低阈值。
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍,将低阈值乘以3作为高阈值。
    zhan = []  # 创建一个空列表，用于存储边缘点的坐标。
    for i in range(1, img_yizhi.shape[0]-1):  # 遍历图像的每一行（除了边界）,外圈不考虑了.
        for j in range(1, img_yizhi.shape[1]-1):  # 遍历图像的每一列（除了边界）。
            if img_yizhi[i, j] >= high_boundary:  # 如果当前像素的灰度值大于等于高阈值，则认为它是边缘点。
                img_yizhi[i, j] = 255  # 将边缘点的灰度值设置为255（白色）。
                zhan.append([i, j])  # 将边缘点的坐标添加到列表中。
            elif img_yizhi[i, j] <= lower_boundary:  # 如果当前像素的灰度值小于等于低阈值，则认为它不是边缘点。
                img_yizhi[i, j] = 0  # 将非边缘点的灰度值设置为0（黑色）。

    while not len(zhan) == 0:  # 当列表不为空时，执行循环。
        temp_1, temp_2 = zhan.pop()  # 出栈,从列表中弹出一个元素，即一个边缘点的坐标。
        a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]  # 获取以该边缘点为中心的3x3邻域。
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):  # 检查邻域中的每个像素是否为边缘点。
            img_yizhi[temp_1-1, temp_2-1] = 255  # 如果是边缘点，则将其标记为边缘。
            zhan.append([temp_1-1, temp_2-1])  # 进栈,将边缘点的坐标添加到列表中。
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yizhi.shape[0]):  # 遍历图像的每一行。
        for j in range(img_yizhi.shape[1]):  # 遍历图像的每一列。
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:  # 如果当前像素既不是边缘点也不是背景色（0或255）。
                img_yizhi[i, j] = 0  # 将非边缘点的灰度值设置为0（黑色）。

    # 绘图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
