import numpy as np

"""
函数中，x是本层网络的激活值，level就是每个神经元要被丢弃的概率。
"""
# dropout函数的实现
def dropout(x, level):
    # 防呆检查
    if level < 0. or level >= 1:  # level是概率值，必须在0~1之间
        raise ValueError('Dropout level must be in interval [0,1]')
    retain_prob = 1. - level  # 保留的神经元

    # 通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当作抛硬币
    # 硬币 正面的概率为p，n表示每个神经元抛的次数
    # 因为我们每个神经元只需要抛一次就可以了所以 n=1，size参数是我们有多少个硬币（神经元的个数）
    # 即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
    random_tensor = np.random.binomial(n=1, p=retain_prob, size=x.shape)
    print(random_tensor)

    x *= random_tensor
    print(x)
    return x


# 输入数据
x = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
dropout(x, 0.4)  # 0.4是删除40%的神经元
