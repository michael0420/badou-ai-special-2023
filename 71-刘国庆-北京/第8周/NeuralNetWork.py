import numpy as np
import scipy.special


class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络,设置输入层,中间层,和输出层节点数,学习率
        # 输入层节点数
        self.inodes = inputnodes
        # 中间层节点数
        self.hnodes = hiddennodes
        # 输出层节点数
        self.onodes = outputnodes
        # 学习率
        self.lr = learningrate

        # 初始化权重矩阵
        # 一个是wih,表示输入层和中间层节点间链路权重形成的矩阵
        # 一个是who,表示中间层和输出层间链路权重形成的矩阵
        # 使用正态分布随机初始化权重矩阵
        # 初始化输入层到隐藏层之间的权重矩阵
        # np.random.normal：生成符合正态分布的随机数矩阵
        # 0.0：正态分布的均值
        # pow(self.hnodes, -0.5)：正态分布的标准差，与隐藏层节点数相关
        # (self.hnodes, self.inodes)：生成的矩阵形状，隐藏层节点数行，输入层节点数列
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 初始化隐藏层到输出层之间的权重矩阵
        # np.random.normal：生成符合正态分布的随机数矩阵
        # 0.0：正态分布的均值
        # pow(self.onodes, -0.5)：正态分布的标准差，与输出层节点数相关
        # (self.onodes, self.hnodes)：生成的矩阵形状，输出层节点数行，隐藏层节点数列
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 每个节点执行激活函数,得到的结果将作为信号输出到下一层,设置激活函数为sigmoid函数
        self.activate_function = lambda x: scipy.special.expit(x)
        pass  # 初始化方法结束

    def train(self, inputs_list, targets_list):
        # 根据输入的训练数据更新节点链路权重
        # 把inputs_list, targets_list转换成np支持的二维矩阵
        # .T表示做矩阵的转置
        # 转换输入数据为np矩阵并进行转置
        inputs = np.array(inputs_list, ndmin=2).T
        # 转换目标数据为np矩阵并进行转置
        targets = np.array(targets_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activate_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activate_function(final_inputs)
        # 计算误差
        # 计算输出层的误差,即目标值与实际输出值之间的差异
        # output_errors 是一个矩阵,包含了每个输出节点的误差
        output_errors = targets - final_outputs
        # 计算隐藏层的误差,通过反向传播（backpropagation）传递输出层的误差
        # self.who.T 表示输出层到隐藏层权重矩阵的转置
        # output_errors * final_outputs * (1 - final_outputs) 表示输出层的误差信号
        # hidden_errors 是一个矩阵,包含了每个隐藏节点的误差
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重的更新量,然后把更新加到原来链路权重上
        # 更新输出层到隐藏层的权重矩阵
        # self.lr 是学习率,用于控制权重更新的步幅
        # np.dot 是矩阵乘法,用于计算权重更新的量
        # output_errors * final_outputs * (1 - final_outputs) 是输出层的误差信号
        # np.transpose(hidden_outputs) 是隐藏层输出信号的转置
        self.who += self.lr * np.dot(output_errors * final_outputs * (1 - final_outputs), np.transpose(hidden_outputs))
        # 更新输入层到隐藏层的权重矩阵
        # self.lr 是学习率,用于控制权重更新的步幅
        # np.dot 是矩阵乘法,用于计算权重更新的量
        # hidden_errors * hidden_outputs * (1 - hidden_outputs) 是隐藏层的误差信号
        # np.transpose(inputs) 是输入信号的转置
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))
        pass  # 训练方法结束

    def query(self, inputs):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activate_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activate_function(final_inputs)
        # 打印最终输出信号
        print(f"最终输出信号:\n{final_outputs}")
        # 返回最终输出信号
        return final_outputs


# 初始化网络
# 由于一张图片总共有28*28 = 784个数值,因此我们需要让网络的输入层具备784个输入节点
input_nodes = 784  # 输入层节点数
hidden_nodes = 200  # 隐藏层节点数
output_nodes = 10  # 输出层节点数
learning_rate = 0.1  # 学习率
# 创建神经网络实例
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读入训练数据
# open函数里的路径根据数据存储的路径来设定
# 打开训练数据文件
training_data_file = open("dataset/mnist_train.csv", 'r')
# 读取训练数据文件中的所有行
training_data_list = training_data_file.readlines()
# 关闭训练数据文件
training_data_file.close()

# 加入epochs,设定网络的训练循环次数
epochs = 5  # 训练循环次数
for e in range(epochs):
    # 把数据依靠','区分,并分别读入
    for record in training_data_list:
        # 使用逗号分隔数据
        all_values = record.split(',')
        # 数据归一化
        # 将输入数据转换为浮点数类型的np数组，并进行归一化
        # all_values[1:] 表示从索引1开始到列表末尾的所有元素，即去掉第一个元素（图片标签）
        # / 255.0 将像素值缩放到范围 [0, 1]
        # * 0.99 缩放到范围 [0, 0.99]
        # + 0.01 平移到范围 [0.01, 1]
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 创建一个包含全是 0.01 的np数组作为目标（标签）
        # output_nodes 是输出层的节点数，用于确定目标数组的大小
        targets = np.zeros(output_nodes) + 0.01
        # 设置目标值
        targets[int(all_values[0])] = 0.99
        # 使用训练数据进行训练
        n.train(inputs, targets)

# 读入测试数据
# 打开测试数据文件
test_data_file = open("dataset/mnist_test.csv")
# 读取测试数据文件中的所有行
test_data_list = test_data_file.readlines()
# 关闭测试数据文件
test_data_file.close()
# 用于存储模型的预测结果
scores = []

# 对测试数据进行预测
for record in test_data_list:
    # 使用逗号分隔数据
    all_values = record.split(',')
    correct_number = int(all_values[0])  # 获取图片的真实数字
    print(f"该图片对应的数字为:{correct_number}")
    # 预处理数字图片
    # 数据归一化
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    print(f"网络认为图片的数字是:{label}")
    if label == correct_number:
        scores.append(1)  # 如果预测正确,将1添加到scores列表
    else:
        scores.append(0)  # 如果预测错误,将0添加到scores列表
print(f"预测结果:{scores}")  # 输出预测结果的列表

# 计算图片判断的成功率
# 将预测结果的列表转换为np数组
# 这样可以更方便地进行统计和计算
scores_array = np.asarray(scores)
# 计算模型在测试数据上的性能
# scores_array.sum() 计算预测正确的样本数量
# scores_array.size 计算总样本数量
# 将正确预测的样本数量除以总样本数量,得到性能（准确率）并输出
print(f"图片判断的成功率:{scores_array.sum() / scores_array.size}")
