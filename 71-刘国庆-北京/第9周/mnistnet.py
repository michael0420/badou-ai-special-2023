# 导入PyTorch库
import torch
# 导入PyTorch中的神经网络模块
import torch.nn as nn
# 导入PyTorch中的优化器模块
import torch.optim as optim
# 导入PyTorch中的函数模块，包含激活函数等
import torch.nn.functional as F
# 导入PyTorch的视觉库
import torchvision
# 导入PyTorch的变换模块，用于图像预处理和数据增强
import torchvision.transforms as transforms


# 定义一个名为Model的类
class Model:
    # 类的初始化方法，接收三个参数：net（神经网络模型）、cost（损失函数）、optimist（优化器）
    def __init__(self, net, cost, optimist):
        # 将传入的神经网络模型赋值给类的net属性
        self.net = net
        # 调用类的create_cost方法，传入损失函数参数，将返回的损失函数对象赋值给类的cost属性
        self.cost = self.create_cost(cost)
        # 调用类的create_optimizer方法，传入优化器参数，将返回的优化器对象赋值给类的optimizer属性
        self.optimizer = self.create_optimizer(optimist)
        # pass表示占位符，不执行任何具体操作，可以在后续添加其他初始化操作
        pass

    # 类的方法create_cost，用于根据传入的cost参数选择或初始化损失函数对象
    def create_cost(self, cost):
        # 定义一个字典，包含支持的损失函数及其对应的PyTorch损失函数对象
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        # 根据传入的cost参数从字典中获取对应的损失函数对象
        selected_cost = support_cost[cost]
        # 返回损失函数对象
        return selected_cost

    # 类的方法create_optimizer，用于根据传入的optimist参数选择或初始化优化器对象
    def create_optimizer(self, optimist, **rests):
        # 定义一个字典，包含支持的优化器及其对应的PyTorch优化器对象
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        # 根据传入的optimist参数从字典中获取对应的优化器对象
        selected_optim = support_optim[optimist]
        # 返回优化器对象
        return selected_optim

    # 类的方法train，用于训练神经网络模型
    def train(self, train_loader, epoches=3):
        # 遍历指定轮数的训练循环
        for epoch in range(epoches):
            # 初始化累计损失值
            running_loss = 0.0
            # 遍历训练数据加载器
            for i, data in enumerate(train_loader, 0):
                # 获取输入数据和标签
                inputs, labels = data
                # 梯度清零
                self.optimizer.zero_grad()
                # 前向传播 + 反向传播 + 参数优化
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # 累计损失值
                running_loss += loss.item()
                # 每隔100个小批量数据打印一次损失值
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' % (
                    epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        # 输出训练结束信息
        print('Finished Training')

    # 类的方法evaluate，用于评估神经网络模型在测试数据集上的准确率
    def evaluate(self, test_loader):
        # 输出评估信息
        print('Evaluating ...')
        # 初始化正确分类的样本数和总样本数
        correct = 0
        total = 0
        # 设置torch.no_grad()上下文管理器，禁用梯度计算，因为在测试和预测时不需要计算梯度
        with torch.no_grad():
            # 遍历测试数据加载器
            for data in test_loader:
                images, labels = data
                # 前向传播
                outputs = self.net(images)
                # 预测类别
                predicted = torch.argmax(outputs, 1)
                # 统计总样本数和正确分类的样本数
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # 输出网络在测试图像上的准确率
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


# 定义加载MNIST数据集的函数
def mnist_load_data():
    # 定义数据预处理的转换操作，包括将图像转换为张量和归一化操作
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, ], [1, ])])
    # 创建训练数据集，包括下载MNIST数据、应用预处理转换
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 创建训练数据加载器，设置批量大小、是否打乱数据和使用的线程数
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    # 创建测试数据集，包括下载MNIST测试数据、应用预处理转换
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # 创建测试数据加载器，设置批量大小、是否打乱数据和使用的线程数
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    # 返回训练和测试数据加载器
    return trainloader, testloader


# 定义一个继承自torch.nn.Module的MNIST神经网络类
class MnistNet(torch.nn.Module):
    # 构造方法，定义神经网络的层结构
    def __init__(self):
        # 调用父类的构造方法
        super(MnistNet, self).__init__()
        # 定义全连接层，输入维度是28*28，输出维度是512
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        # 定义全连接层，输入维度是512，输出维度是512
        self.fc2 = torch.nn.Linear(512, 512)
        # 定义全连接层，输入维度是512，输出维度是10（对应10个数字类别）
        self.fc3 = torch.nn.Linear(512, 10)

    # 前向传播方法，定义数据在网络中的流动过程
    def forward(self, x):
        # 将输入张量展平为一维张量，大小为28*28
        x = x.view(-1, 28 * 28)
        # 第一个全连接层，使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 第二个全连接层，使用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 第三个全连接层，使用Softmax激活函数，输出预测概率分布
        x = F.softmax(self.fc3(x), dim=1)
        # 返回网络的输出
        return x


# 如果该脚本作为主程序运行
if __name__ == '__main__':
    # 创建MNIST神经网络模型
    net = MnistNet()
    # 创建包含神经网络、损失函数（交叉熵）和优化器（RMSprop）的模型
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    # 加载MNIST训练和测试数据
    train_loader, test_loader = mnist_load_data()
    # 使用训练数据对模型进行训练
    model.train(train_loader)
    # 使用测试数据对模型进行评估
    model.evaluate(test_loader)
