import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 归一化三种方式


# 1、公式(x-μ)/σ
def z_score(data):
    # 求μ，σ
    μ = np.mean(data)
    σ = sum((x - μ) ** 2 for x in data) / len(data)
    return [(x - μ) / σ for x in data]


# 2、公式(x-min)/(max-min)   (0, 1)
def normalization1(data):
    return [(x - min(data)) / (max(data) - min(data)) for x in data]


# 3、公式(x-mean)/(max-min)   (-1, 1)
def normalization2(data):
    return [(x - np.mean(data)) / (max(data) - min(data)) for x in data]


if __name__ == "__main__":
    # 输入20个0-100的数
    # input_random = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
    #      11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

    input_random = sorted(np.random.randint(0, 10, size=(20, 1)))

    output_z_score = z_score(input_random)
    output_norm1 = normalization1(input_random)
    output_norm2 = normalization2(input_random)

    output_count = []
    # output_count = [list(input_random).count(x) for x in input_random]
    for i in input_random:
        c = input_random.count(i)
        output_count.append(c)


    plt.title("标准归一化")
    plt.plot(input_random, output_count)
    plt.plot(output_z_score, output_count)
    plt.show()
    plt.title("(0, 1)归一化")
    plt.plot(input_random, output_count)
    plt.plot(output_norm1, output_count)
    plt.show()
    plt.title("(-1, 1)归一化")
    plt.plot(input_random, output_count)
    plt.plot(output_norm2, output_count)
    plt.show()

