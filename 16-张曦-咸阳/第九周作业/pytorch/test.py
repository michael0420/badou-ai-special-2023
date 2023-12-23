import torch

a = torch.tensor([[2], [3], [4]])  # 三行一列
print(a)
b = torch.tensor([[2, 2, 2], [3, 3, 3], [5, 5, 5]])  # 三行两列
print(b.size())

c = a.expand_as(b)  # 3行3列 expand_as 把a扩展为类似b size 的矩阵
print(c)
print(c.size())


tensor_one = torch.tensor([[1,1],[2,2]])
tensor_two = torch.tensor([[1,2],[2,3]])

# 会生成一个布尔张量，其中每个元素是模型预测的类别是否与真实标签相匹配。
# 例如，如果一个样本被正确预测，则对应的元素值为 True，否则为 False。
result = tensor_one == tensor_two

print("result = ", result)
