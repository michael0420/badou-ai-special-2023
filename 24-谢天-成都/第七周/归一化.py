import numpy as np
import matplotlib.pyplot as plt


#1
def Normalization1(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]
#2
def Normalization2(x):
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]

#标准化
def z_score(x):
    x_mean=np.mean(x)
    s2=sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    return [(i-x_mean)/s2 for i in x]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1=[]
cs=[]

for i in l:
    c=l.count(i)
    cs.append(c)
print(cs)

g1=Normalization1(l)
g2=Normalization2(l)
g3=z_score(l)
print(g1)
print(g2)
print(g3)

plt.plot(l,cs)
plt.plot(g1,cs)
plt.plot(g2,cs)
plt.plot(g3,cs)

plt.show()
