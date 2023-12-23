import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_loss(cost)
        self.optimizer = self.create_optimizer(optimist)

    def create_loss(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optimizer = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optimizer[optimist]

    def train(self, train_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0

            """enumerate(train_loader, 0)
            返回一个迭代器，它产生(i, data)
            的元组，其中
            i
            是迭代的索引，data
            是一个
            mini - batch
            的输入和标签。"""
            for i, data in enumerate(train_loader, 0):
                # 获取输入数据和标签
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.net(inputs)

                # 计算损失
                loss = self.cost(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss %.3f' % (
                        epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))

                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print("Evaluating....")
        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("accuracy of the network on the test images: %d %%" % (100 * correct / total))


def mnist_load_data():
    """
    transforms.ToTensor() 将图像数据从PIL Image对象或NumPy数组转换为PyTorch的Tensor对象。
    将像素值从范围 [0, 255] 缩放到范围 [0.0, 1.0]。具体地说，每个像素值都会被除以 255.0。

    transforms.Normalize(mean, std) 用于标准化图像数据
    在这个例子中，[0,] 表示均值，[1,] 表示标准差。这是因为Normalize期望的输入是一个均值和一个标准差的列表，每个通道一个值。
    该转换将每个通道的值减去均值，然后除以标准差，以使数据分布在接近零的范围内。

    最终，通过transforms.Compose() 将这两个转换组合在一起，形成一个转换的序列。
    这个组合可以用于数据加载器（例如torchvision.datasets.ImageFolder）中，在加载图像时对其进行预处理。

    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0, ], [1, ])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, testloader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
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
