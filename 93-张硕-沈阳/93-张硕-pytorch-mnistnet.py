import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
'''
1、定义好网络
2、编写数据的标签和索引路径
3、把数据送入网络
'''

#定义训练
class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

#定义损失函数
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

#定义优化器
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]
#训练
    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss =0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad() #清零梯度

                outputs = self.net(inputs) #正向传播
                loss = self.cost(outputs, labels) #计算损失函数
                loss.backward() #反向传播梯度计算
                self.optimizer.step() #使用优化器更新参数

                running_loss += loss.item()
                if i % 100 ==0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('Finished Trainning')

#推理
    def evaluate(self, test_loader):
        print('Evaluating..................')
        correct = 0
        total = 0
        with torch.no_grad(): #不计算梯度
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)

                predicted = torch.argmax(outputs, 1)
                #print('####predicted/n',predicted)
                #print('####labels\n', labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


#处理数据
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
'''
torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
这段代码创建了一个数据加载器，
用于从testset数据集中加载数据，
每个批次包含32个样本，
数据将在每个训练周期开始时被打乱，
并使用2个子进程来加载数据。
'''


#定义网络
class MnisNet(torch.nn.Module):
    def __init__(self):
        super(MnisNet, self).__init__()
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
    net = MnisNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)




