import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


# 加载mnist数据集
def mnist_load_data():
    # 数据转换
    # 将多个transform组合起来
    transform = transforms.Compose([
        # 
        transforms.ToTensor(),
        # 传入均值，方差
        transforms.Normalize([0, ], [1, ])
    ])
    # 获取训练集
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 加载训练集
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    # 获取测试集
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # 加载测试集
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, testloader


# 定义神经网络,需要继承nn.Module
class MnistNet(torch.nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__()
        # 定义输入层到隐含层
        self.layer1 = torch.nn.Linear(28 * 28, 512)
        # 隐含层到下一个隐含层
        self.layer2 = torch.nn.Linear(512, 512)
        # 定义隐含层到输出层
        self.layer3 = torch.nn.Linear(512, 10)

    # 继承后需要定义前向传播方法   
    def forward(self, x):
        # 传入的是图像像素矩阵，需要转为一行
        # view函数会影响原始张量
        x = x.view(-1, 28 * 28)
        # 
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # 最后输出层用softmax激活函数
        x = F.softmax(self.layer3(x), dim=1)
        return x


# 定义模型类
class Model:
    def __init__(self, network, cost, optim, lr=0.1):
        # 定义神经网络
        self.network = network
        # 学习率
        self.lr = lr
        # 定义损失函数
        self.cost = self.create_cost(cost)
        # 定义优化器
        self.optimizer = self.create_optimizer(optim)

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': torch.nn.CrossEntropyLoss(),
            'MSE': torch.nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimizer(self, optimizer, **rests):
        support_optim = {
            'SGD': torch.optim.SGD(self.network.parameters(), lr=self.lr, **rests),
            'ADAM': torch.optim.Adam(self.network.parameters(), lr=self.lr, **rests),
            'RMST': torch.optim.RMSprop(self.network.parameters(), lr=self.lr, **rests)
        }
        return support_optim[optimizer]

    def train(self, train_loader, epoches=3):
        # 循环训练多少轮
        for epoch in range(epoches):
            # 定义保存loss变量
            running_loss = 0.0
            # 遍历训练数据
            for i, data in enumerate(train_loader):
                inputs, labels = data
                # 梯度归零
                self.optimizer.zero_grad()
                # 前向传播计算预测值
                outputs = self.network(inputs)
                # 计算loss
                loss = self.cost(outputs, labels)
                # 反向传播计算梯度
                loss.backward()
                # 使用优化器更新参数
                self.optimizer.step()
                # 保存loss
                running_loss += loss.item()
                # 每100次打印一下loss
                if i % 100 == 0:
                    print('当前第%d轮，第%d次，训练了%.2f%%, loss:%.3f' % (
                          (epoch + 1, i, (i + 1) * 1. / len(train_loader), running_loss / 100)))
                    running_loss = 0.0
        print('训练完成,保存model')

    def evaluate(self, test_loader):
        print('开始测试推理')
        correct = 0
        total = 0
        # 设置模型不计算梯度
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                # 前向传播，计算输出值
                outputs = self.network(images)
                # 获取输出值最大值索引
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('准确度%.2f' % (100 * correct / total))


if __name__ == '__main__':
    # 初始化神经网络
    network = MnistNet()
    # 获取数据集加载器
    train_loader, test_loader = mnist_load_data()
    print(len(train_loader))
    print(len(test_loader))
    # 初始化模型
    model = Model(network, 'CROSS_ENTROPY', 'SGD', 0.1)
    # 模型训练
    model.train(train_loader)
    # 模型预测
    model.evaluate(test_loader)
