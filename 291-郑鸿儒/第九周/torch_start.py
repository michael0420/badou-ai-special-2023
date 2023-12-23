import torch
import torch.nn as nn


# model = nn.Sequential()
# model.add_module('fc1', nn.Linear(3, 4))
# model.add_module('fc2', nn.Linear(4, 5))
#
# model = nn.Sequential(
#     nn.Conv2d(1, 2, 5),
#     nn.ReLU(),
# )
#
# model = nn.ModuleList([nn.Conv2d(1, 2, 5), nn.ReLU()])

# x = torch.randn([3, 4], requires_grad=True)
# y = 2 * x
# y_sum = torch.sum(y)
#
# print(y_sum.requires_grad)
# y_sum.backward()
# print(x.grad)

# class Linear(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(Linear, self).__init__()
#         # 使用nn.Parameter包裹，告诉torch这是一个模型参数，需要梯度更新
#         self.weight = nn.Parameter(torch.randn(in_features, out_features))
#         if bias:
#             self.bias = self.weight = nn.Parameter(torch.randn(out_features))
#
#     def forward(self, x):
#         x = x.mm(self.weight)
#         if self.bias:
#             x = x + self.bias.expand_as(x)
#         return x
#
#
# net = Linear(3, 4)
# x = net.forward()
# print('111', x)
