import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms



class Model:
    def __init__(self, cost_type, optim_type, epochs):
        self.net = Net()
        self.epochs = epochs

        self.cost_dict = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        self.optim_dict = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001)
        }
        self.optim = self.optim_dict[optim_type]
        self.cost = self.cost_dict[cost_type]

    def train(self, data_train):
        # 循环每个epoch进行训练，计算损失，反向传播
        for epoch in range(self.epochs):
            total_loss = 0.0
            for index, data in enumerate(data_train, 0):
                inputs, labels = data

                self.optim.zero_grad()

                predict = self.net(inputs)
                loss = self.cost(predict, labels)
                loss.backward()
                self.optim.step()

                total_loss += loss.item()
                # 每100个epoch打印一次平均损失
                if index % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (index + 1)*1./len(data_train), total_loss / 100))
                    total_loss = 0.0

        print('训练结束')

    def test(self, data_test):
        print('开始测试')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_test:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('测试集结果: %d %%' % (100 * correct / total))


# 数据读取
def dataloader():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0, ], [1, ])
        ]
    )

    data_train = torchvision.datasets.MNIST(root='./data', train=True, download='True', transform=transform)
    train_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=2)

    data_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(data_test, batch_size=32, shuffle=True, num_workers=2)

    return train_loader, test_loader


# 训练的分类网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_block1 = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=True),
        nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=True),
        nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        feature = self.conv_block1(x)
        feature2 = self.conv_block2(feature)
        b, c, h, w = feature2.shape
        a = feature2.squeeze(dim=1)
        predict = self.fc(feature2.squeeze(dim=1).view(-1, h * w))

        return predict


if __name__ == "__main__":
    # 自定义训练模型，三个超参，损失类型+优化器类型+训练次数
    model = Model('CROSS_ENTROPY', 'RMSP', 10)
    # 加载torch自带的mnist训练集和测试集
    train_data, test_data = dataloader()
    # 训练
    model.train(train_data)
    # 测试
    model.test(test_data)


