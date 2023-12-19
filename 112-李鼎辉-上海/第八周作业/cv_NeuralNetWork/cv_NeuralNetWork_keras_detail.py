import numpy
import scipy.special
class NeuralNetWork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        # 输入层
        self.inodes = inputnodes
        # 隐藏层
        self.hnodes = hiddennodes
        # 输出层
        self.onodes = outputnodes
        # 学习率
        self.lr = learningrate
        # 初始化输入层到隐藏层的权重矩阵
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
        # 初始化隐藏层到输出层的权值矩阵
        self.who = (numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes)))
        # 定义激活函数（Sigmoid函数）
        # 输入x： 输入是神经元的加权输入，即输入信号与对应权重的加权和。
        # 输出f(x)： 输出是经过 Sigmoid 函数处理后的结果，范围在 0 到 1 之间。
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    # 训练
    '''
    inputs_list:神经网络的输入数据，一组特征值
    target_list:神经网络的目标数据，样本的标签或者期望的输出值
    '''
    def train(self,inputs_list,target_list):
        # 将numpy数组转换为二维再进行转置。（特征数，样本数）
        inputs = numpy.array(inputs_list, ndmin=2).T
        # （输出节点数，样本数）
        targets = numpy.array(target_list, ndmin=2).T

        # 正向传播
        # 输入层到隐藏层的权重矩阵与转换为（特征数，样本数）的矩阵进行点积操作
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隐藏层的输入经过激活函数的变换得到隐藏层的输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 隐藏层到输出层的权重矩阵和隐藏层的输出做点积
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 激活函数sigmoid
        final_outputs = self.activation_function(final_inputs)

        # 计算实际输出与目标输出的误差
        output_errors = targets-final_outputs
        # 计算隐藏层的误差，通过将输出层的误差通过权重矩阵传播回隐藏层。
        hidden_errors = numpy.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))

        # 更新权重，新权值=当前权值-学习率x梯度
        self.who += self.lr*numpy.dot((output_errors*final_outputs*(1-final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr*numpy.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)), numpy.transpose(inputs))

        pass
    # 查询
    def query(self, inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs
# 输入节点
input_nodes = 784
# 隐藏节点
hidden_nodes = 200
# 输出节点
output_nodes = 10
# 学习率
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open('dataset/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]))/255.0*0.99+0.01
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs, targets)
test_data_file = open('dataset/mnist_test.csv')
test_data_list =  test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    inputs = (numpy.asfarray(all_values[1:]))/255.0*0.99+0.01

    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print("网络认为图片的数字是:",label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
scores_array = numpy.asarray(scores)
print("perfermance=", scores_array.sum()/scores_array.size)

import numpy as np

# 一维数组的点积
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result_1d = np.dot(a, b)
print("1D Dot Product:", result_1d)

# 二维数组的点积
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result_2d = np.dot(A, B)
print("2D Dot Product:\n", result_2d)
