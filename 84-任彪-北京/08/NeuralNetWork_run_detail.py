import numpy
import matplotlib.pyplot as plt
import scipy.special

[1]
#open函数里的路径根据数据存储的路径来设定
data_file = open("dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()
print(len(data_list))
print(data_list[0])

#把数据依靠','区分，并分别读入
all_values = data_list[0].split(',')
#第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))

#最外层有10个输出节点
onodes = 10
targets = numpy.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99
print(targets)  #targets第8个元素的值是0.99，这表示图片对应的数字是7(数组是从编号0开始的).

[2]
'''
根据上述做法，我们就能把输入图片给对应的正确数字建立联系，这种联系就可以用于输入到网络中，进行训练。
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点。
这里需要注意的是，中间层的节点我们选择了100个神经元，这个选择是经验值。
中间层的节点数没有专门的办法去规定，其数量会根据不同的问题而变化。
确定中间层神经元节点数最好的办法是实验，不停的选取各种数量，看看那种数量能使得网络的表现最好。
'''
#初始化网络
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3


class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate
        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        '''
        scipy.special.expit对应的是sigmod函数.
        lambda是Python关键字，类似C语言中的宏定义，当我们调用self.activation_function(x)时，编译器会把其转换为spicy.special_expit(x)。
        '''
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        # 根据输入的训练数据更新节点链路权重
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        oj = numpy.transpose(hidden_outputs)

        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    def query(self, inputs):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print("最后的预测值",final_outputs)
        return final_outputs


n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#读入训练数据
#open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv")
trainning_data_list = training_data_file.readlines()
training_data_file.close()
#把数据依靠','区分，并分别读入
for record in trainning_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
    #设置图片与数值的对应关系
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

[3]
'''
最后我们把所有测试图片都输入网络，看看它检测的效果如何
'''
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:",correct_number)
    #预处理数字图片
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    #让网络判断图片对应的数字,推理
    outputs = n.query(inputs)
    #找到数值最大的神经元对应的 编号
    label = numpy.argmax(outputs)  
    print("output reslut is : ", label)
    #print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)

[4]
'''
在原来网络训练的基础上再加上一层外循环
但是对于普通电脑而言执行的时间会很长。
epochs 的数值越大，网络被训练的就越精准，但如果超过一个阈值，网络就会引发一个过拟合的问题.
'''
#加入epocs,设定网络的训练循环次数
epochs = 10

for e in range(epochs):
    #把数据依靠','区分，并分别读入
    for record in trainning_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

import cv2

img = cv2.imread("dataset/my_own_2.png")
img_1 = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), (28,28), interpolation=cv2.INTER_CUBIC)
height, width, _ = img.shape
pixels = []
for y in range(height):
    for x in range(width):
        # 获取像素值
        pixel = img_1[y, x]
        # 将像素值添加到集合中
        pixels.append(pixel)
queryData = numpy.ravel(pixels)
final_outputs = n.query(queryData)

