import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
#处理数据
def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0, ], [1, ])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    return train_loader, test_loader
class net(nn.Module):
    def __init__(self):
        #初始化，将需要变的参数放在这里
        super(net, self).__init__()
        self.f1 = torch.nn.Linear(28*28,512)
        self.f2 = torch.nn.Linear(512,512)
        self.f3 = torch.nn.Linear(512,10)
    def forward(self,x):
        #不需要变的函数及参数
        x = x.view(-1, 28*28)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.softmax(self.f3(x), dim=1)
        return x

class model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]
    def train(self,train_loader, epoches=3):
        for epoch in range(epoches):
            thisloss = 0.0
            for i,data in enumerate(train_loader,0):
                input,label = data

                #初始化梯度，归零
                self.optimizer.zero_grad()

                '''
                forward + backward +optimize
                '''
                #forward
                outputs = self.net.forward(input)
                #backward
                loss = self.cost(outputs,label)
                loss.backward()
                # optimize
                self.optimizer.step()
                #将张量取出数
                thisloss += loss.item()
                #每一百次输出一次平均误差
                if i%100 ==0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), thisloss / 100))
                    thisloss = 0.0
        print('finished ')
    def evaluate(self,test_loader):
        #记录
        corr = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                image,labels = data

                outputs = self.net(image)
                pre = torch.argmax(outputs,1)
                total += labels.size(0)
                corr += (pre == labels.sum().item())
        print('Accuracy of the network on the test images: %d %%' % (100 * corr / total))



if __name__ == '__main__':
    net = net()
    model = model(net,'CROSS_ENTROPY','RMSP')
    train_loader,test_loader = load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
