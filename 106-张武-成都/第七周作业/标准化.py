#pylint:disable=E0602
'''
归一化2种方式
0-1之间
x = (x - x_min)/(x_max - x_min)

-1到1之间
x = (x - x_mean)/(x_max - x_min)

标准化 z_score 
x = (x-μ)/σ

'''
import numpy as np
import matplotlib.pyplot as plt



# 0-1之间
# x = (x - x_min)/(x_max - x_min)

def normalizetion1(x):
    return [(float(i) - min(x))/(max(x)-min(x)) for i in x]
    
# -1到1之间
# x = (x - x_mean)/(x_max - x_min)
def normalizetion2(x):
    return [(float(i) - np.mean(x))/(max(x)-min(x)) for i in x]
    
# 标准化 z_score 
# x = (x-μ)/σ    
def z_score(x):
    mean = np.mean(x)
    sigma = np.sqrt(np.sum([(i -mean)**2 for i in x])/len(x))
    return [(i-mean)/sigma for i in x]
    

m = np.random.randint(10,50,20)
print('随机数', m)

n1 = normalizetion1(m)
print('0到1', n1)

n2 = normalizetion2(m)
print('-1到1', n2)

n3 = z_score(m)
print('标准化', n3)

