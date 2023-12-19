import numpy as np
import matplotlib.pyplot as plt
import random
#归一化
# x = (x-xmean)/(xmax-xmin)
def N1(x):
    xmean = np.mean(x)
    xmax  = max(x)
    xmin  = min(x)
    xmax_min = xmax - xmin
    return [(i-xmean)/xmax_min for i in x]

# x = (x - xmin)/(xmax - xmin)
def N2(x):
    xmax =max(x)
    xmax_min = xmax - min(x)
    return [(i-xmax)/xmax_min for i in x]

#标准化
# x = (x - xmean)/σ
def z_score(x):
    xmean = np.mean(x)
    Dt = np.std(x)
    return [(i - xmean)/Dt for i in x]

xx = [random.randint(-10,10) for i in range(100)]
Y1 = N1(xx)
Y2 = N2(xx)
Y3 = z_score(xx)

plt.figure(11)
plt.subplot(2,1,1)
plt.plot(xx)
plt.subplot(2,1,2)
plt.plot(Y1)
plt.plot(Y2)
plt.plot(Y3)
plt.legend(['Y1','Y2','Y3'])
plt.figure(22)
xx = sorted(xx)
cs = []
for i in xx:
    c = xx.count(i)
    cs.append(c)
plt.plot(xx,cs)
Y1 = sorted(Y1)
plt.plot(Y1,cs)
plt.show()

