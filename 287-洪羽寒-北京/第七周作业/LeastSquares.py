import pandas as pd

csv = open('E:\\自动备份文档\\Python\\修改配置\\修改配置.csv','r',encoding='gbk')

sales=pd.read_csv('train_data.csv',sep='\s*,\s*',engine='python')
X=sales['X'].values
Y=sales['Y'].values

s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = 4

#循环累加
for i in range(n):
    s1 = s1 + X[i]*Y[i]
    s2 = s2 + X[i]
    s3 = s3 + Y[i]
    s4 = s4 + X[i]*X[i]

#计算斜率和截距
k = (s2*s3-n*s1)/(s2*s2-s4*n)
b = (s3 - k*s2)/n                               #y=1.4x+3.5
print(k, b)
print("Coeff: {} Intercept: {}".format(k, b))
