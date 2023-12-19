"""
手写代码实现手写数字识别
"""

import numpy as np
import scipy.special

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes,learningrate):
        #初始化网络，设置输入层、输出层和中间层
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #学习率
        self.learn = learningrate

        #初始化权重矩阵
        #wih为输入层和中间层节点间权重形成的矩阵；who为中间层和输出层节点间权重形成矩阵
        self.wih = (np.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes, self.inodes)))
        self.who = (np.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes,self.hnodes)))
        """每个节点执行激活函数，得到的结果作为信号输出到下一层，使用sigmoid作为激活函数"""
        #scipy.special.expit为sigmoid函数，匿名函数用于对输入x进行激活
        self.activation_function = lambda x : scipy.special.expit(x)

        pass

    """根据输入的训练数据更新节点链路权重"""
    def train(self, inputs_list, targets_list):

        #将输入和输出矩阵转化为二维矩阵，转置后为n*1维
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #计算输入层到中间层的信息
        hidden_inputs = np.dot(self.wih, inputs)

        #隐藏层输出，sigmoid激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        #输出层接受来自中间层的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        #输出层激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)

        #计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))

        #更新权重
        #np.transpose转换数组形状：m*n->n*m
        self.who += self.learn * np.dot((output_errors*final_outputs*(1-final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.learn * np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),np.transpose(inputs))
        
        pass

    def query(self, inputs):
        #计算中间层从输入层接手到的信息量
        hidden_inputs = np.dot(self.wih, inputs)
        #计算中间层经过激活函数形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算输出层接收到的信息量
        final_inputs = np.dot(self.who, hidden_outputs)
        #计算输出层经过激活函数后的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

#一张图片有28*28=784个数值，需要让输入层具备784个输入结点   
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learn_rate = 0.1
net = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learn_rate)

#读入训练数据
train_data_file = open("dataset/mnist_train.csv","r") #100个
train_data_list = train_data_file.readlines()
train_data_file.close()

#加入epocs，设定网络的循环次数
epochs = 5

for i in range(epochs):
    for record in train_data_list:
        #每个图片像素值成列表
        all_values = record.split(",")
        #np.asfarray()转化成浮点型数组
        #防止出现大量为0的重复情况
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片和数值的对应关系，设置标签
        #每个record数组的标签：对应的数字位置设置成0.99，其它设置成0.01
        targets = np.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        #遍历训练，一个一个训练
        net.train(inputs, targets)

test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []
for record in test_data_list:
    all_values = record.split(",")
    correct_number = int(all_values[0])
    print("该图片对应的数字为：",correct_number)

    #预处理图片
    inputs = (np.asfarray(all_values[1:]))/255.0*0.99+0.01
    #判断对应的数字
    outputs = net.query(inputs)
    #找到数字最大的神经元对应的编号
    label = np.argmax(outputs)

    print("网络认为图片的数字为：",label)

    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)

print("scores in test_data:",scores)

#判断图片预测正确的成功率
scores_array = np.asfarray(scores)
print("performance:",scores_array.sum()/scores_array.size)
