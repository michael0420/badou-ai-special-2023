import numpy as np
import matplotlib.pyplot as plt

def Normal1(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

def Normal2(x):
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

def z_(x):
    x_mean=np.mean(x)
    s=sum([(i-x_mean)**2 for i in x])/len(x)
    return [(i-x_mean)/s for i in x]

ls=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

cs=[]

for i in ls:
    c=ls.count(i)
    cs.append(c)

n=Normal1(ls)
n1=Normal2(ls)
z=z_(ls)

print(n)
print(n1)
print(z)

plt.plot(n,cs,"b")
plt.plot(n1,cs,'g')

plt.plot(z,cs,'r')#0均值归一

plt.show()
