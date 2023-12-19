import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    '''
    torch.nn（神经网络）是 PyTorch 中用于构建神经网络模型的核心模块。这个模块提供了许多类和函数
    torch.nn 模块中一些重要的组件：
          神经网络模块 (torch.nn.Module):
          各种层（Layers）:(torch.nn.Linear)、卷积层 (torch.nn.Conv2d)、循环神经网络层 (torch.nn.RNN) 等
          激活函数（Activation Functions）:  ReLU (torch.nn.ReLU)、Sigmoid (torch.nn.Sigmoid)、Tanh (torch.nn.Tanh) 
          损失函数（Loss Functions）: 均方误差损失 (torch.nn.MSELoss)、交叉熵损失 (torch.nn.CrossEntropyLoss) 
          数据转换（Data Transformations）:  于数据转换的函数，如池化、激活函数等。
    '''
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    '''
    
    torch.optim 是 PyTorch 中用于实现优化算法的模块。
    这个模块提供了各种常见的优化算法，如梯度下降、Adam、RMSprop 等，以及一些调整学习率的方法。主要的类和函数包括：

        优化器类 (Optimizer classes):
        
        torch.optim.SGD: 随机梯度下降（Stochastic Gradient Descent）优化器。
        torch.optim.Adam: Adam 优化器。
        torch.optim.RMSprop: RMSprop 优化器。
    '''
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            '''
            enumerate 函数返回一个枚举对象，该对象包含一个索引计数和可迭代对象中的元素
            '''
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                '''
                清零梯度： 将之前计算得到的梯度值归零。这是因为 PyTorch 默认会累积（accumulate）梯度，而不是每次都重置梯度。

                防止梯度累积： 在进行反向传播和梯度计算时，梯度值会累积在模型的参数上
                如果不清零梯度，下一次反向传播时，新计算得到的梯度将会加到之前的梯度上，导致梯度的不正确累积。
                '''
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # 正向传播
                outputs = self.net(inputs)
                # 计算损失函数
                loss = self.cost(outputs, labels)
                # 反向传播：用于计算模型参数梯度的过程，以便更新模型参数以最小化损失函数。
                loss.backward()
                # 参数更新：通过梯度下降的方式更新模型的参数，以减小损失函数。
                self.optimizer.step()
                # 统计新的loss
                running_loss += loss.item()
                # 统计每代 迭代后的速度和损失函数
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return trainloader, testloader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
