import numpy as np

def normalize(data):
    # 计算数据的最小值和最大值
    min_val = np.min(data)
    max_val = np.max(data)

    # 归一化数据
    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data
def mean_normalize(data):
    # 计算数据的均值
    mean_val = np.mean(data)

    # 均值归一化数据
    normalized_data = data - mean_val

    return normalized_data



def z_score_normalize(data):
    # 计算数据的均值和标准差
    mean_val = np.mean(data)
    std_val = np.std(data)

    # Z-score 归一化数据
    normalized_data = (data - mean_val) / std_val

    return normalized_data

# 示例数据
example_data = np.array([1, 2, 3, 4, 5 ,6 ])


# 归一化数据
normalized_data= normalize(example_data)
# 均值归一化数据
normalized_data1 = mean_normalize(example_data)
# Z-score 归一化数据
normalized_data2 = z_score_normalize(example_data)

print("原始数据:", example_data)
print("归一化后的数据:", normalized_data)
print("原始数据:", example_data)
print("均值归一化后的数据:", normalized_data1)
print("Z-score 归一化后的数据:", normalized_data2)