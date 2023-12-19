"""
pytorch对手写数字的识别
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

class Model:
    def __init__(self, net, cost, optimist):
        #网络
        self.net = net
        #损失函数
        self.cost = self.create_cost(cost)
        #优化器
        self.optimizer = self.create_optimizer(optimist)
        pass
    
    #创建可选损失函数
    def create_cost(self, cost):
        support_cost = {
            "CROSS_ENTROPY": nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]
    
    #创建可选优化器
    def create_optimizer(self, optimist, ** rest):
        support_optim = {
            "SGD":optim.SGD(self.net.parameters(),lr=0.1, **rest),
            "ADAM":optim.Adam(self.net.parameters(), lr=0.01, **rest),
            "RMSP":optim.RMSprop(self.net.parameters(), lr=0.001, **rest)
        }

        return support_optim[optimist]
    
    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            #0作为起始索引值，从0开始对train_loader迭代
            for i,data in enumerate(train_loader,0):
                #输入和标签
                inputs, labels = data #图片数据,Inputs维度【图片个数，1，图像宽，图片长】
                
                #将优化器梯度清零，避免梯度积累
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.cost(outputs,labels)
                loss.backward()

                #通过step()更新网络参数，使得损失函数最小化
                self.optimizer.step()

                running_loss += loss.item()

                #100一次
                if i%100 == 0:
                    # print("[epoch %d, %.2f%%] loss: %.3f" %
                    #       (epoch+1, (i+1)*1.0/len(train_loader), running_loss/100))
                    running_loss = 0.0
        print("Finished Training")

    def evaluate(self, test_loader):
        print("Evaluating...")
        #记录正确样本数和总样本数
        correct = 0
        total = 0

        #使用torch.no_grad()上下文管理器
        with torch.no_grad():
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs,1)
                total += labels.size(0) #读取一次里面的图片个数
                correct += (predicted == labels).sum().item()
        print("Accuracy:", correct, total)

    def predict_digit(self, test_image):
        print("Predicting...")
        with torch.no_grad():
            outputs = self.net(test_image)
            predicted = torch.argmax(outputs,1)
        return predicted


#加载数据集并返回训练数据和测试数据加载器

def mnist_load_data():

    #定义预处理
    transform = transforms.Compose(
        [transforms.ToTensor(), #将图像转换为张量
         #减去均值并除以标准差，从而使数据的均值为0，标准差为1
         transforms.Normalize([0,],[1,])
         ]
    )

    #分别创建训练集和测试集的数据集对象
    trainset = torchvision.datasets.MNIST(root="./data", train=True,
                                          download=True, transform = transform)
    testset = torchvision.datasets.MNIST(root="./data", train=False,
                                         download=True, transform=transform)
    
    #分别创建训练集和测试集的数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    tetsloader = torch.utils.data.DataLoader(testset,batch_size=32,
                                             shuffle=True, num_workers=2)
    return trainloader, tetsloader

#定义神经网络模型
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        #定义3个全连接层
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512,512)
        self.fc3 = torch.nn.Linear(512,10)

    def forward(self,x):
        x = x.view(-1, 28*28)
        #进入网络层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    
if __name__ == "__main__":
    net = MnistNet()
    model = Model(net, "CROSS_ENTROPY","RMSP")
    #导入数据
    train_loader, test_loader = mnist_load_data()
    print("train_loade.size:",len(train_loader))
    print("test_loader.size:",len(test_loader))
    model.train(train_loader)
    model.evaluate(test_loader)

    # 图像预处理
    
    # target = Image.open('target2.png').convert('L')
    # plt.imshow(target)
    # plt.show()
    # transform = transforms.Compose([
    # transforms.ToTensor()
    # ])
    # target = transform(target).unsqueeze(0)

    # target_figure = model.predict_digit(target)
    # print("target_figure:",target_figure)

