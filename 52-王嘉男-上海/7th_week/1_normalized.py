import numpy as np


# 示例数据
example_data = np.array(
    [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30])

# 归一化数据
normalized_data = (example_data - np.min(example_data)) / (np.max(example_data) - np.min(example_data))
# 均值归一化数据
normalized_data1 = example_data - np.mean(example_data)
# Z-score 归一化数据
normalized_data2 = (example_data - np.mean(example_data)) / np.std(example_data)

print("原始数据:", example_data)
print("归一化后的数据:", normalized_data)
print("原始数据:", example_data)
print("均值归一化后的数据:", normalized_data1)
print("Z-score 归一化后的数据:", normalized_data2)
