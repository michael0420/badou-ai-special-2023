import numpy as np
import scipy.special


class NeuralNetWork:

    def __init__(self, input_nodes, head_nodes, output_nodes, learningrate):
        self.inodes = input_nodes
        self.hnodes = head_nodes
        self.onodes = output_nodes

        self.lr = learningrate

        # 随机初始化输入到隐藏权重、隐藏到输出权重
        # self.hide_weight = np.random.rand(self.inodes, self.hnodes) - 0.5
        self.hide_weight = (np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes,self.inodes)))
        # self.out_weight = np.random.rand(self.hnodes, self.onodes) - 0.5
        self.out_weight = (np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes,self.hnodes)))

        # 构造激活函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        """
        根据输入训练数据更新节点链路权重
        :param inputs_list:  输入的训练数据
        :param targets_list: 训练数据对应的正确结果。
        第一步是计算输入训练数据，给出网络的计算结果
        第二步是将计算结果与正确结果相比对，获取误差，采用误差反向传播法更新网络里的每条链路权重。
        """
        # 输入值
        inputs = np.array(inputs_list, ndmin=2).T
        # 标准答案
        targets = np.array(targets_list, ndmin=2).T
        # 输入到隐藏数据矩阵乘
        input_hidden = np.dot(self.hide_weight, inputs)
        # 过隐藏激活函数
        hidden_activate = self.activation_function(input_hidden)
        # 隐藏到输出层
        hidden_output = np.dot(self.out_weight, hidden_activate)
        # 输出层过激活
        out_activate = self.activation_function(hidden_output)

        # 计算误差
        output_err = targets - out_activate
        # 隐藏层误差
        hidden_err = np.dot(self.out_weight.T, out_activate*(1-out_activate) * output_err)

        # 新权值 = 当前权值 - 学习率 × 梯度
        self.out_weight += self.lr * np.dot(output_err * out_activate * (1 - out_activate), np.transpose(hidden_activate))
        self.hide_weight += self.lr * np.dot(hidden_err * hidden_activate * (1 - hidden_activate), np.transpose(inputs))
        pass

    def query(self, inputs):
        # 根据输入的测试数据计算最佳答案
        # 计算输入数据经过隐藏层之后的数据
        inputs_hidden = np.dot(self.hide_weight, inputs)
        # 经过激活函数
        hidden_output = self.activation_function(inputs_hidden)

        # 隐藏层->输出层权重乘上层的输出
        final_input = np.dot(self.out_weight, hidden_output)

        # 经过输出激活函数最终输出
        final_outputs = self.activation_function(final_input)

        print(final_outputs)
        return final_outputs



#初始化网络
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# open函数里的路径根据数据存储的路径来设定
data_file = open("dataset/mnist_train.csv")
train_data_list = data_file.readlines()
data_file.close()
print(len(train_data_list))
print(train_data_list[0])

epochs = 5
for e in range(epochs):
    for record in train_data_list:
        all_values = record.split(",")
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01

        # 构造labes
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)


test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []


for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:",correct_number)
    # 预处理数字图片
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算图片判断的成功率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)