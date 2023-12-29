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

    def create_cost(self, cost):
        support_cost = {#（损失函数）
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):#深度学习优化算法
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)#一次正向
                loss = self.cost(outputs, labels)#计算loss损失
                loss.backward()#反向，计算梯度更新权重
                self.optimizer.step()#调用优化

                running_loss += loss.item()
                if i % 500 == 0:
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
    def __init__(self):#init放可更新参数的内容
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)#fc full connect全连接
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc4 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)

    def forward(self, x):#forward放不需要更新参数的内容，并整合init的内容
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc4(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

if __name__ == '__main__':
    # train for mnist
    net = MnistNet()#构建网络
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')#构建包括损失函数:CROSS_ENTROPY,优化算法:RMSP的模型
    train_loader, test_loader = mnist_load_data()#加载手写数字数据
    model.train(train_loader,epoches=5)#训练
    model.evaluate(test_loader)#测试及推理
