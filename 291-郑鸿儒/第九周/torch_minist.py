import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


class Module:
    def __init__(self, net, cost, optimizer):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimizer)
        self.cost_1 = cost
        self.optimizer_1 = optimizer

    def create_cost(self, cost):
        support_cost = {
            "mse": nn.MSELoss(),
            "cross_entropy": nn.CrossEntropyLoss()
        }
        return support_cost[cost]

    def create_cost_1(self):
        support_cost = {
            "mse": nn.MSELoss(),
            "cross_entropy": nn.CrossEntropyLoss()
        }
        return support_cost[self.cost_1]

    def create_optimizer(self, optimizer, **rest):
        support_optimizer = {
            "SGD": optim.SGD(self.net.parameters(), lr=0.1, **rest),
            "Adam": optim.Adam(self.net.parameters(), lr=0.01, **rest),
            "RMSP": optim.RMSprop(self.net.parameters(), lr=0.001, **rest)
        }
        return support_optimizer[optimizer]

    def create_optimizer_1(self):
        support_optimizer = {
            "SGD": optim.SGD(self.net.parameters(), lr=0.1),
            "Adam": optim.Adam(self.net.parameters(), lr=0.01),
            "RMSP": optim.RMSprop(self.net.parameters(), lr=0.001)
        }
        return support_optimizer[self.optimizer_1]

    # 魔法代码,第一遍写的时候loss死活不能降,第二遍按老师的写正常,第三遍换回自己写的又好了
    # 真他妈的玄学,这TM甚至都不配称为换了写法,艹
    def train(self, train_loader, epochs=3):
        for e in range(epochs):
            # batch loss
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # 一个batch的数据
                inputs, labels = data
                # 优化器
                optimizer = self.create_optimizer_1()
                # 损失函数
                cost = self.create_cost_1()
                # 优化器梯度归零
                optimizer.zero_grad()
                # self.optimizer.zero_grad()
                # 预测值 对于继承nn.module 的类(网络),创建之后直接调用等同于forward
                # 即net() <=> net.forward()
                outputs = self.net(inputs)
                # 计算loss
                loss = cost(outputs, labels)
                # loss = self.cost(outputs, labels)
                # 反向传播
                loss.backward()
                # 更新权重
                self.optimizer.step()
                # 累加running_loss
                running_loss += loss.item()
                if (i + 1) % 100 == 0:
                    print(
                        "当前进度第%d代第%d轮已进行 %.2f%%, loss: %.3f" %
                        (e + 1, i + 1, float(i + 1) / len(train_loader) * 100, running_loss / 100)
                    )
                    running_loss = 0.0
        print("train finish")

    def evaluate(self, test_loader):
        print("start evaluating...")
        correct = 0
        total = 0
        # 预测不需要梯度!!!!!
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                total += labels.size(0)
                outputs = self.net(inputs)
                predict_res = torch.argmax(outputs, 1)
                correct += (predict_res == labels).sum().item()
        print("共%d组数据, 正确率 %.3f%%" % (total, float(correct) / total * 100))


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.Fc1 = nn.Linear(28 * 28, 512)
        self.Fc2 = nn.Linear(512, 512)
        self.Fc3 = nn.Linear(512, 10)

    def forward(self, x):
        # x.view 有返回值,不会改变原值
        x = x.view(-1, 28*28)
        x = nn.functional.relu(self.Fc1(x))
        x = nn.functional.relu(self.Fc2(x))
        # 在dim=1维度计算softmax，默认dim=None,计算全部softmax
        x = nn.functional.softmax(self.Fc3(x), dim=1)
        return x


def mnist_load():
    # 创建数据处理管道
    transform = transforms.Compose([
        # 将图像数据转换乘torch tensor
        transforms.ToTensor(),
        # 标准化（归一化） 均值0，方差1
        transforms.Normalize([0, ], [1, ])
    ])
    trainset = torchvision.datasets.MNIST(root="data", train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=5)
    testset = torchvision.datasets.MNIST(root="data", train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=5)

    return trainloader, testloader


if "__main__" == __name__:
    mnistnet = MnistNet()
    train_loader, test_loader = mnist_load()
    model = Module(mnistnet, "cross_entropy", "RMSP")
    model.train(train_loader)
    model.evaluate(test_loader)
