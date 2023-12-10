import numpy
import numpy as np
import scipy

class NeuralNetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 输入层节点数=图像大小
        self.input_nodes = input_nodes
        # 隐含层节点数
        self.hidden_nodes = hidden_nodes
        # 输出层节点数=图像类别数
        self.output_nodes = output_nodes
        # 学习率
        self.lr = learning_rate
        # 激活函数 sigmoid 公式: 1 / 1 + e^(-x)
        self.activation_function = lambda x: scipy.special.expit(x)
        # 损失函数 - 均方误差 mse 公式:
        # self.loss_function = ''
        # 优化器

        # 权重随机初始化
        self.wih = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.who = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

    def train(self, trains, labels):
        # 开始训练
        # 转成numpy格式的二维矩阵
        inputs = np.array(trains, ndmin=2).T    # shape = (784,1)
        targets = np.array(labels, ndmin=2).T   # shape = (10,1)
        # 前向传播 - 计算预测值
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.wih, inputs)    # (200, 781) * (781, 1) = (200, 1)
        # 过激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs)     # (10, 200) * (200, 1) = (10, 1)
        # 过激活函数
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        # 输出层误差 = 正确结果与节点输出结果的差值
        output_errors = targets - final_outputs
        # 隐含层误差
        # 节点的激活函数，所有输入该节点的链路把经过其上的信号与链路权重做乘积后加总，再把加总结果进行激活函数运算;
        # 再乘以权重(绿色部分)
        # (200, 10) * (10, 1) = (200, 1)
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))

        # 更新权重
        # 更新隐含层到输出层的权重
        # (10 * 1) * (1, 200) = (10, 200)
        self.who += self.lr * numpy.dot(output_errors * final_outputs * (1 - final_outputs), np.transpose(hidden_outputs))
        # (200, 1) * (1, 784) = (200, 784)
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))

    def query(self, inputs):
        # 推理
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        # 过激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 过激活函数
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# 读取训练数据和测试数据
training_data_file = open('dataset/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 初始化模型
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# y = model.query([1.0, 0.5, 1.5])
# print(y)

# 总共训练多少轮
epochs = 50
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        x_train = np.asfarray(all_values[1:])
        x_label = all_values[0]
        # 数据归一化
        # 第一列为y真实值, 剔除第一列, 取出样本数据, 并转为浮点数类型
        inputs = x_train / 255.0 * 0.99 + 0.01
        # one hot
        targets = np.zeros(output_nodes) + 0.01
        targets[int(x_label)] = 0.99
        # 模型训练
        n.train(inputs, targets)

print(n.who)
print(n.wih)

# 测试模型
test_data_file = open('dataset/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []
for record in test_data_list:
    all_values = record.split(',')
    x_train = np.asfarray(all_values[1:])
    correct_number = int(all_values[0])
    # 数据归一化
    # 第一列为y真实值, 剔除第一列, 取出样本数据, 并转为浮点数类型
    inputs = x_train / 255.0 * 0.99 + 0.01
    # 模型推理
    outputs = n.query(inputs)
    # 找出最大值索引
    label = np.argmax(outputs)
    print('模型推理结果: ', correct_number, label, correct_number == label)
    if correct_number == label:
        scores.append(1)
    else:
        scores.append(0)

print(scores)

# 计算准确度
score_array = np.asarray(scores)
print('准确度=', score_array.sum() / score_array.size)
