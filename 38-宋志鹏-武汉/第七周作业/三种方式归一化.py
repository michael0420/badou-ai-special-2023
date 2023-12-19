import numpy as np

# 1、最大最小归一化，将所有特征缩放到（0，1）范围内，公式：（x-min(x)）/(max(x)-min(x))
def min_max_normalion(data):
    x = (data-np.min(data)) / (np.max(data) -  np.min(data))
    return x

data1 = np.array([1,2,3,4,5])
normallized_data1 = min_max_normalion(data1)
print ('最大最小归一化：\n',normallized_data1)



# 2、Z-score标准化归一化，数据会根据其均值（μ）和标准差（σ）进行归一化，公式：(x - μ) / σ
def z_score_normalize(data):
    x = (data - np.mean(data)) / np.std(data)
    return x

data2 = np.array([1,2,3,4,5])
normallized_data2 = z_score_normalize(data2)
print ('Z-score标准化：\n',normallized_data2)



# 3、L2归一化
'''
# L2归一化会调整数据，使得每个样本的L2范数（即各元素的平方和的平方根）为1。这在处理文本或图像数据时非常有用。
# 范数：在数学中是一个用来衡量向量大小的函数，通常表示为，其中x是一个向量。
# L2范数：向量中每个元素的平方和的平方根，即∥x∥₂=√(x₁²+x₂²+...+xᵢ²)，其中i表示向量的维数。
# 在机器学习和数据处理中，通常会使用范数对数据进行归一化，以便不同的特征或样本在量级上具有可比性，
    并且可以避免特征之间的权重差异过大对结果造成影响。
'''
def l2_normalize(data):
    l2_norm = np.linalg.norm(data,ord=2)
    x = data /l2_norm
    return x

data3 =  np.array([1,2,3,4,5])
normallized_data3 = l2_normalize(data3)
print ('L2归一化：\n', normallized_data3)