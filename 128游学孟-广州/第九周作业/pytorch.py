import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image

# 设置随机种子
torch.manual_seed(42)

#数据预处理和加载
"""
transforms.ToTensor()：将数据转换为张量
transfor.Normailize((0.1307,),(0.3081,)):对图像数据进行标准化处理。
这里使用了MNIST数据集在训练集上计算得到的均值喝标准差进行标准化。
datasets.MNIST函数中，train-True表示加载训练集，train-False表示加载测试集
download=True表示如果本地不存在该苏剧集，则会自动从互联网下载。
torch.utils.data.DataLoader：用于创建数据加载器。
数据加载器是一个可以迭代访问数据集的对象，可以方便地将数据分成小批次进行训练。其中，batch_size指定每个批次的样本数量，
shuffle=True表示在每个epoch开始时对数据进行洗牌，以增加数据的随机性。
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# 构建模型
"""
在__init__方法中，我们使用super(Net,self).__init__()调用父类的构造函数，以确保基类的初始化得到执行。
定义三个连接层（线性层）:
self.fc1:(784,128)
self.fc2:(128,64)
self.fc3:(64,10)
forward:定义了前向传播过程：x.view(-1,784)操作转换为二维张量，-1表示自动计算维度。

"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net() # 创建实例

# 定义损失函数和优化器
"""
nn.CrossEntropyLoss()是交叉熵损失函数的实现。
optim.SGD()是随机梯度下降法的实现。调用model.parameters()可以获取模型中所有需要进行更新的参数。
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 训练模型
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# 模型评估
"""
model.eval()表示将模型设置为评估模式。
test_loss = 0 用于记录测试集上的累计损失值。
correct = 0 用于记录模型在测试集上预测正确的样本数量。
with torch.no_grad():表示进入一个上下文环境，在该环境中不会进行梯度计算。
for data,target in test_loader:遍历测试数据集中的所有数据。
output = model(data)利用当前模型对输入数据进行前向传播，得到模型的输出结果。
test_loss += criterion(output,target).item()计算模型输出结果与标签之间的损失值，并累加到test_loss变量中
pred = output.argmax(dim=1,keepdim = True)根据模型的输出结果，取出每个样本预测的最大值所对应的类别标签。
correct += pred.eq(target.view_as(pred)).sum().item()将预测结果与真实标签进行比较，统计预测正确的样本数量，并累加到correct变量中。
test_loss /=len(test_loader.dataset)计算平均测试损失值，即将累计损失值除以测试数据集的大小。
accuract = 100.*correct/len(test_loader.dataset)计算模型在测试数据集上的准确率。
"""
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

def predict(photo):
    # 加载并预处理图片
    image = Image.open(photo).convert('L')  # 转换为灰度图像
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # 调整大小为28x28像素

            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.1307,), (0.3081,))  # 标准化处理
        ]
    )
    input_tensor = transform(image).unsqueeze(0)  # 添加批次维度

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        # 打印预测结果
        print("Predicted digit:", predicted.item())





# 训练和测试模型
for epoch in range(1, 11):
    photo = "D://xuexi//zuoye//week9//2.jpg"
    train(epoch)
    test()
    predict(photo)






