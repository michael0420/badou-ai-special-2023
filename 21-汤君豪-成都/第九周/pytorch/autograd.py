import torch

x = torch.randn((4, 4), requires_grad=True)
print(x)
y = 2*x
z = y.sum()
print(x)
print(y)
print(z)

z.backward()
print(x.grad)
'''
tensor([[ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.]])
'''