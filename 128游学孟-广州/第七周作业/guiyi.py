import numpy as np
from sklearn.preprocessing import  MinMaxScaler,StandardScaler

# 生成随机数据
data = np.random.rand(5,3)
print("原始数据：\n",data)

# 最小-最大归一化
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)
print("\n最小-最大归一化后的数据：\n",data_minmax)

#z-score归一化
standard_scaler = StandardScaler()
data_zscore = standard_scaler.fit_transform(data)
print("\nz-score归一化后的数据：\n",data_zscore)

# L2范数归一化
data_l2 = data / np.linalg.norm(data,ord = 2,axis =1,keepdims=True)
print("\nL2范数归一化后的数据：\n",data_l2)