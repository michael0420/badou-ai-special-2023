import numpy as np
import random
import pylab
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['FangSong'] #横众轴显示字体为‘仿宋’的中文标签
plt.rcParams['axes.unicode_minus']=False

def ransac(data, model_net, iteration, threshold, ratio):
    total, _ = data.shape
    # 每次迭代随机选取的内群点的个数
    group_num = int(total * ratio)
    max_num = 0
    best_k, best_b = 0, 0
    real_group_points_index = []
    # 迭代开始
    while iteration > 0:
        num = 0
        tmp_index = []
        # 1、随机选取内群点的索引
        group_index = random.sample(range(0, total), group_num)
        # 通过索引获得内群点的值
        group_points = data[group_index]

        # 2、用最小二乘法计算内群的模型，得到参数k和b
        k, b = model_net.get_k_b(group_points)

        # 3、用内群模型计算非内群模型的损失
        for index in range(total):
            # 不计算假设的内群点
            if index not in group_index:
                loss = model_net.get_loss(data[index, :], k, b)
                if loss < threshold:
                    # 4、记下非假设的内群点中满足阈值(满足阈值就是内群点)的数量
                    num += 1
                    tmp_index.append(index)
        # 5、比较哪次计算中内群数量最多,内群最多的那次所建的模型就是我们所要求的解
        if num > max_num:
            max_num = num
            best_k, best_b = k, b
            real_group_points_index = tmp_index
        iteration -= 1

    return best_k, best_b, real_group_points_index


# 设计最小二乘法模型，只要两个部分即可，一个确定直线，一个是计算损失，
class model_least_square:
    def get_k_b(self, all_data):
        data_x = all_data[:, 0]
        data_y = all_data[:, 1]
        total = len(data_x)
        a, b, c, d = 0, 0, 0, 0
        for i in range(total):
            a += data_x[i] * data_y[i]
            b += data_x[i]
            c += data_y[i]
            d += data_x[i] ** 2
        k = (b * c - total * a) / (b ** 2 - d * total)
        b = (c - k * b) / total
        return b, k

    def get_loss(self, data, k, b):
        data_x = data[0]
        data_y = data[1]
        loss = ((k * data_x + b) - data_y) ** 2
        return loss


def prepare(x_range, k_range, points_num, offset_points_num, offset):
    # 准备数据
    # 准备points_num个点的x，每个点都在0-x_range上
    points_x = np.random.randint(0, x_range, size=(points_num, 1))
    # 生成一个随机的标准斜率作为答案，这个斜率限制在0到k_range之间
    standard_k = np.random.randint(0, k_range)
    # 计算出所有点的y
    points_y = points_x * standard_k

    # 加入高斯偏移,在原有坐标下对x和y加入一个默认以0为均值，方差为5的随机数，这个随机数基本是-5-+5之间
    noise_points_x = points_x + 5 * np.random.normal(0, 5, size=points_x.shape)
    noise_points_y = points_y + 5 * np.random.normal(0, 5, size=points_x.shape)

    # 修改正常点中的offset_points_num个点，将offset_points_num个点设置为离群点
    # 随机生成offset_points_num个不重复的索引，修改这offset_points_num个索引对应的坐标值,x值就不改了，y都随机加-offset到+offset
    solitude_points_index = random.sample(range(0, points_num), offset_points_num)
    # noise_points_x[solitude_points_index] += 10 * np.random.randint(1, 2)
    # noise_points_y[solitude_points_index] += 10 * np.random.randint(1, 2)
    # noise_points_y[solitude_points_index] += offset * np.random.normal(size=(1, 1))
    noise_points_y[solitude_points_index] += offset * np.random.randint(5, 10)

    # 得到所有非噪声点
    standard_points_index = []
    for index in range(points_num):
        if index not in solitude_points_index:
            standard_points_index.append(index)

    all_points = np.hstack((noise_points_x, noise_points_y))

    return all_points, noise_points_x, noise_points_y, standard_points_index, solitude_points_index, standard_k


if __name__ == "__main__":

    # 准备数据集
    # prepare函数参数
    # 输入超参数
    # x_range(横坐标0-x_range范围内随机生成点), k_range(标准斜率在0-k_range随机生成),
    # points_num(生成总点数), offset_points_num(偏离点的个数), offset(偏离点的偏移量)
    # 输出超参数
    # all_points(所有原始生成点的坐标(x, y)，都在标准直线有= kx上),
    # noise_points_x(高斯偏移后所有点的横坐标), noise_points_y(高斯偏移后所有点的纵坐标),
    # standard_points_index(正常点的序号), solitude_points_index(偏移点的序号), standard_k(随机的标准斜率)
    all_points, noise_points_x, noise_points_y, standard_points_index, solitude_points_index, standard_k = prepare(200, 2, 500, 40, 10)

    # 实例化最小二乘模型
    model = model_least_square()
    # 利用最小二乘计算模型
    least_square_k, least_square_b = model.get_k_b(all_points)

    # 利用ransac计算模型
    # data(数据), model_net(最小二乘模型), k(最大迭代次数), threshold(阈值), ratio(内群点比例)
    best_k, best_b, real_group_points_index = ransac(all_points, model, 500, 7e3, 0.2)

    # 绘制图像
    # 绘非制噪声点
    pylab.plot(noise_points_x[standard_points_index], noise_points_y[standard_points_index], 'k.', label='噪声点')
    # 绘制噪声点
    pylab.plot(noise_points_x[solitude_points_index], noise_points_y[solitude_points_index], 'r.', label='原始点')
    x = np.linspace(0, 100, 100)
    # 绘制标准答案曲线
    pylab.plot(x, standard_k * x, label='标准')
    # 绘制最小二乘结果曲线
    pylab.plot(x, least_square_k * x + least_square_b, label='最小二乘结果')
    # 绘制RANSAC结果曲线
    pylab.plot(x, best_k * x + best_b, label='RANSAC结果')

    pylab.legend()
    pylab.show()


