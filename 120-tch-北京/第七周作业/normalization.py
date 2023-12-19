import numpy as np
import math

def Normalization1(x):

    nor1 = [(float(i) - min(x))/(max(x)-min(x)) for i in x]
    nor2 = [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]
    return nor1,nor2


def z_score(x):

    square_deviation = sum([(i - np.mean(x))*(i - np.mean(x)) for i in x])/len(x)  #for i in x 这里[]容易漏掉,其应该放在整个式子的最后
    stantard_deviation = math.sqrt(square_deviation)
    z = [(i - np.mean(x))/stantard_deviation for i in x]

    return z

l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

nor1,nor2 = Normalization1(l)
std = z_score(l)

print('0-1归一化：',nor1)
print('-1-1归一化：',nor2)
print('标准化：',std)

