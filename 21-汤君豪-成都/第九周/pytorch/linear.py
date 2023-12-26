import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        print(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.randn(in_features))
            print(self.bias)

    def forward(self, x):
        x = x.mm(self.weight)
        print(x)
        if self.bias != None:
            x += self.bias.expand_as(x)
        return x

if __name__ == "__main__":
    net = Linear(3, 2)
    y = net.forward(torch.tensor([[2., 1.], [3., 2.], [4., 3.]]))
    print(y)
