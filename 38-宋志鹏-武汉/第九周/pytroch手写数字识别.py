import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Model:
    # 初始化网络、损失函数和优化器
    def __init__(self,net,cost,optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    # 根据传入的字符串选择损失函数
    def create_cost(self,cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    # 根据传入的字符串和其他参数选择优化器
    def create_optimizer(self, optimist, **rests):

        support_optim = {
            'SGD': optim.SGD(self.net.parameters(),lr=0.1,**rests),
            'ADAM': optim.Adam(self.net.parameters(),lr=0.01,**rests),
            'RMSP': optim.RMSprop(self.net.parameters(),lr=0.001,**rests)
        }
        return support_optim[optimist]

    # 进行训练，其中使用梯度下降和反向传播
    def train(self,train_loader,epoches=3):
        for epoch in range(epoches):
            # 初始化一个变量running_loss为0，用于记录每个轮次的累计损失
            running_loss = 0.0
            for i ,data in enumerate(train_loader,0):

                # 将data解包为输入数据inputs和标签labels
                inputs,labels = data

                # 将模型的梯度缓冲区清零，以便进行下一次前向传播和反向传播
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # 将输入数据inputs通过模型self.net进行前向传播，得到模型的输出outputs
                outputs = self.net(inputs)
                loss = self.cost(outputs,labels)
                # 根据计算图进行反向传播，计算模型参数的梯度
                loss.backward()
                # 根据计算得到的梯度更新模型的参数
                self.optimizer.step()

                # 累加损失值
                running_loss += loss.item()
                if i % 100 == 0:
                    # 输出当前轮次、进度百分比和损失值
                    print('[epoch %d ,%.2f%%] loss: %.3f' %
                          (epoch + 1,(i + 1)*1./len(train_loader),running_loss /100))
                    # 将running_loss重置为0，用于下一次累计损失计算
                    running_loss = 0.0

        print ('Finished Training')

    # 评估模型准确率
    def evaluate(self,test_loader):
        print('Evaluating ...')
        # 初始化记录预测正确的样本数量
        correct = 0
        # 初始化记录测试样本的总数
        total = 0

        with torch.no_grad():
            for data in test_loader:
                images,labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs,1)
                total += labels.size(0)
                '''将预测正确的样本数量累加到correct变量中。使用(predicted == labels)
                进行元素比较得到一个布尔值的张量，然后使用.sum()
                计算所有预测正确的样本数量，最后使用.item()
                将结果转换为Python整数类型'''
                correct += (predicted == labels).sum().item()

        # 输出模型在测试图像上的准确率，使用百分号格式化输出
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# 加载MNIST数据集并返回训练集和测试集的数据加载器
def mnist_load_data():
    # 将数据转换为张量，并对数据进行归一化处理
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,],[1,])])

    # 进行数据转换。train = True表示加载训练集,train=False表示加载测试集
    # 通过指定batch_size为32，shuffle = True进行数据打乱，
    # num_workers = 2使用两个进程进行数据加载
    trainset = torchvision.datasets.MNIST(root='./data',train=True,
                                          download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,
                                            shuffle=True,num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data',train=False,
                                         download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=32,
                                             shuffle=True,num_workers=2)
    return trainloader,testloader



class MnistNet(torch.nn.Module):
    '''
    在初始化方法__init__中，首先调用super函数来初始化父类torch.nn.Module。
    然后定义了三个全连接层：self.fc1、self.fc2和self.fc3。
    这里的torch.nn.Linear表示一个线性层，前一个数字表示输入特征的数量，
    后一个数字表示输出特征的数量
    '''
    def __init__(self):
        super(MnistNet,self).__init__()
        self.fc1 = torch.nn.Linear(28*28,512)
        self.fc2 = torch.nn.Linear(512,512)
        self.fc3 = torch.nn.Linear(512,10)
    '''在前向传播方法forward中，输入x首先通过x.view(-1,28*28)进行形状变换，
        将输入的2D图像展平为1D向量。
        然后通过F.relu函数将输入传入三个全连接层，并通过激活函数ReLU进行非线性变换。
    '''
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

if __name__== '__main__':
    net = MnistNet()
    model = Model(net,'CROSS_ENTROPY','RMSP')
    train_loader,test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)