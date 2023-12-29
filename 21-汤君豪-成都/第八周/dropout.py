import numpy as np

# 自造数据
data = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

# dropout函数实现
# 函数中, x是本层网络的激活值。Level就是每个神经元要被丢弃的概率。
def dropout(x, level):
    if level < 0. or level > 1:
        raise ValueError('Dropout level must be in interval [0, 1].')

    retain_prob = 1 - level
    # 通过binomial函数，生成与x一样的维数向量。binomial函数就像抛使币一样，我们可以把每个神经元当做抛硬币一样
    # 硬币 正面的概率为p，n表示每个神经元试验的次数
    # 因为我们每个神经元只需要抛一次就可以了，所以n=1，size参数是我们有多少个硬币。
    random_tensor = np.random.binomial(n=1, p=retain_prob, size=x.shape)
    new_x = x * random_tensor

    return new_x

new_data = dropout(data, 0.4)
print(new_data)