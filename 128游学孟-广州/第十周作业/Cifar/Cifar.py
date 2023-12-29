import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import multiprocessing as mp

# 定义网络模型
"""
使用简单的卷积神经网络模型，包含两个全连接层和池化层。
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) #定义了一个卷积层，输入通道数为3，输出通道数为16，卷积核大小为3，填充为1
        self.relu = nn.ReLU() # 定义了一个ReLU激活函数
        self.maxpool = nn.MaxPool2d(2, 2) #定义了一个最大池化层，池化核大小为2，步长为2。
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) #定义第二个卷积层，输入通道数为16，输出通道数为32，卷积核大小为3，填充为1.
        self.fc1 = nn.Linear(32 * 8 * 8, 64) #定义了一个全连接层，输入特征数为32*8*8，输出特征数为64
        self.fc2 = nn.Linear(64, 10)#定义了另外一个全连接层，输入特征数为64，输出特征数为10（对应着CIFAR-10数据集的类别数）

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载CIFAR-10数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义训练函数
def train(net, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0 # 初始化一个变量running_loss，用于记录每个小批次的损失之和。
        for i, data in enumerate(trainloader, 0): # 这个循环用于遍历训练集的小批次数据，enumerate(trainloader,0)返回的是一个索引-数据对的迭代器，i是索引，data是一个包含输入和标签的数据对。

            inputs, labels = data # 将输入和标签从数据对中分离出来。

            optimizer.zero_grad() # 清零优化器的梯度信息，以便进行新一轮的反向传播和参数更新。

            outputs = net(inputs)
            loss = criterion(outputs, labels) #计算模型在当前小批次上的损失，使用指定的损失函数cruterion对模型的输出和真实标签进行比较
            loss.backward()  #执行反向传播，计算损失相对于模型参数的梯度。
            optimizer.step() #根据梯度更新模型的参数，执行优化步骤。

            running_loss += loss.item() #将当前小批次的损失值累加到running_loss变量中。
            if i % 200 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

# 定义测试函数
def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy on test set: %.2f %%' % accuracy)

if __name__ == '__main__':
    mp.set_start_method('spawn')

    # 创建网络实例并训练、测试模型
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(net, criterion, optimizer, epochs=5)
    test(net)
