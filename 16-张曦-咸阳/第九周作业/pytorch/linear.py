import torch


class Linear(torch.nn.Module):  # 继承nn.Module
    def __init__(self, in_feature, out_feature):
        super(Linear, self).__init__()  # 等价于nn.Module.__init__(self)

        # torch.randn 用来生成随机数字的tensor，这些随机数字满足标准正态分布（0~1）
        self.weight = torch.nn.Parameter(torch.randn(out_feature, in_feature))
        self.bias = torch.nn.Parameter(torch.randn(out_feature))

    def forward(self, x):
        x = x.mm(self.weight)

        # expand_as 作用是将输入bias的维度扩展为与指定x相同的size
        x = x + self.bias.expand_as(x)
        return x


if __name__ == '__main__':
    # train for min set
    net = Linear(3, 2)
    x = net.forward
    print(x)
