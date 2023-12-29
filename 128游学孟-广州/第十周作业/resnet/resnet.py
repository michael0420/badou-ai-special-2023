import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms

#使用预训练的ResNet-50模型进行微调的分类模型，其中ResNet-50模型的全连接层被替换为适应特定类别数的全连接层
"""
初始化网络模型
@:param
self.resnet:Resnet-50模型定义，self.resnet()也是正向传播函数.
num_features:最后一次全连接层的输入特征
self.resnet.fc(num_features,num_classes),num_features为最后一次全连接层，num_classes适应类别数。
"""
class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()

        self.resnet = models.resnet50(pretrained=True) #创建一个预训练的Resnet-50模型
        num_features = self.resnet.fc.in_features  # 获取ResNet-50模型最后一次全连接层的输入特征。
        self.resnet.fc = nn.Linear(num_features, num_classes) # 将RetNet-50模型的全连接层替换为新的连接层，适应特定类别数

    def forward(self, x):
        x = self.resnet(x)
        return x

"""
模型训练
model:这是要训练的模型，通常是一个神经网络模型。在寻来你过程中，我们将通过优化损失函数来更新模型的参数，使其能够更好地拟合训练数据。
train_loader:这是一个数据加载器，用于从训练数据集中批量加载数据。数据加载器负责将数据划分为小批次，并在每个训练迭代中提供了一批数据样本给模型进行训练。
optimizer:优化器用于更新模型参数的算法。它根据计算得到的梯度信息调整模型的权重以最小化损失函数。
criterion:损失函数是评估模型预测结果与真实标签之间差异的函数。在训练过程中，我们使用损失函数来度量模型的预测与真实标签之间的误差，并通过最小化误差来优化模型。
device:设备参数制定了训练过程中使用的计算设备，例如CPU或GPU。将数据和模型参数移动到适当的设备上可以加速计算并提高训练效果。
通常使用torch.device来指定设备，如device = torch.device("cuda" if torch.cuda.is_available() else "cpu")。
"""
def train(model, train_loader, optimizer, criterion, device):
    model.train() # 将模型设置为训练模式
    running_loss = 0.0 # 初始化损失值
    # 遍历训练数据加载器
    for images, labels in train_loader:
        # 将输入数据和标签移动到指定设备上
        images, labels = images.to(device), labels.to(device)

        # 清零参数梯度
        optimizer.zero_grad()

        # 前向传播:计算模型的输出
        outputs = model(images)

        # 计算损失值
        loss = criterion(outputs, labels)

        # 反向传播:计算梯度并更新模型参数
        loss.backward()
        optimizer.step()

        # 累加损失值
        running_loss += loss.item()

    # 返回平均损失值
    return running_loss / len(train_loader)

"""
:param在训练模型时，要使用训练集对模型进行训练，并使用验证集对模型进行调参。为了真正测试模型的性能，需要将其应用于测试集并计算其准确率。
evaluate():这个函数用来评估模型在测试集上的性能表现的。
model:训练好的模型。被用于进行前向传播，即将输入数据传入模型并生成输出。模型应该是一个能够接收输入数据并返回输出的Pytorch模型对象。
test_loader:测试集数据加载器。它是一个PyTorch的DataLoader对象，用于批量加载测试集数据。一次性将整个测试加载到内存中可能导致内存不足。
使用数据加载器来按批次加载数据，以提高内存效率。
criterion:损失函数。常用的损失函数包括交叉熵损失函数，均方误差损失函数。
device:设备类型
for images,labels in test_loader
在每次迭代中，test_loader会返回一个包含图像数据和标签的批次。这个批次由images和labels两个变量接收
images:这是一个张量，包含了当前批次中的图像数据。它的形状通常为[batch_size,channels,height,width],其中batch_size表示批次大小，channels表示通道数
height和width分别表示图像的高度和宽度。
labels:这是一个张量，包含了当前批次中的真实标签数据。它的形状通常为[batch_size]，其中batch_size 表示批次大小。每个元素对应一个样本的真实标签。
"""

def evaluate(model, test_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0 # 初始化运行损失
    correct = 0 # 初始化正确预测数
    total = 0 # 初始化总样本呢数

    with torch.no_grad(): #禁用梯度计算，提高运行效率
        for images, labels in test_loader: # 遍历测试数据集数据加载器中的每个批次
            images, labels = images.to(device), labels.to(device) # 将输入数据和标签移动到指定设备上

            outputs = model(images) # 使用模型进行前向传播，得到输出
            loss = criterion(outputs, labels)  #计算模型输出和真实标签之间的损失

            running_loss += loss.item() # 累加运行损失

            _, predicted = torch.max(outputs.data, 1) # 找到每个样本中最大值以及其位置
            total += labels.size(0) #累加总样本数
            correct += (predicted == labels).sum().item() # 计算正确预测的样本数

    accuracy = 100 * correct / total # 计算准确率
    return running_loss / len(test_loader), accuracy #返回损失和准确率


# 定义超参数和其他设置
num_classes = 10 # 类别数
learning_rate = 0.001 # 学习率
batch_size = 128 # 批次大小
num_epochs = 10 #训练轮数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 设置设备（GPU或CPU）

# 加载数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True) #训练集
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor()) #测试集

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 训练数据加载器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # 测试数据加载器

# 初始化模型、损失函数和优化器
model = ResNetModel(num_classes).to(device) #初始化模型
criterion = nn.CrossEntropyLoss() # 定义交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # 定义Adam优化器，用于更新模型参数

# 训练和评估模型
for epoch in range(num_epochs):
    # 训练模型
    train_loss = train(model, train_loader, optimizer, criterion, device) #调用train函数进行模型训练，并计算训练损失（train_loss）

    # 评估模型
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device) # 调用train函数进行模型训练，并计算训练损失（train_loss）

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
