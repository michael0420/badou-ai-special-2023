import numpy
import scipy.special

class  NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):        #初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes


        self.lr = learningrate                                                     #设置学习率

        self.wih = (numpy.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes,self.inodes) )  )      #-0.5向下取整
        self.who = (numpy.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes,self.hnodes) )  )

        self.activation_function = lambda x:scipy.special.expit(x)        #每个节点执行激活函数，得到的结果将作为信号输出到下一层，我们用sigmoid作为激活函数
        
        pass
        
    def  train(self,inputs_list, targets_list):

        inputs = numpy.array(inputs_list, ndmin=2).T                         #根据输入的训练数据更新节点链路权重
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)                          #计算信号经过输入层后产生的信号量
        hidden_outputs = self.activation_function(hidden_inputs)             #中间层神经元对输入的信号做激活函数后得到输出信号
        final_inputs = numpy.dot(self.who, hidden_outputs)                   #输出层接收来自中间层的信号量
        final_outputs = self.activation_function(final_inputs)               #输出层对信号量进行激活函数后得到最终输出信号


        output_errors = targets - final_outputs                              #计算误差
        hidden_errors = numpy.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))        #根据误差计算链路权重的更新量，然后把更新加到原来链路权重上

        self.who += self.lr * numpy.dot((output_errors * final_outputs *(1 - final_outputs)),
                                       numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                       numpy.transpose(inputs))
                                       
        pass
        
    def  query(self,inputs):                                            #根据输入数据计算并输出答案
        hidden_inputs = numpy.dot(self.wih, inputs)                     #计算中间层从输入层接收到的信号量
        hidden_outputs = self.activation_function(hidden_inputs)        #计算中间层经过激活函数后形成的输出信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)              #计算最外层接收到的信号量
        final_outputs = self.activation_function(final_inputs)          #计算最外层神经元经过激活函数后输出的信号量
        print(final_outputs)
        return final_outputs
        
#初始化网络
input_nodes = 784                                               #由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("dataset/mnist_train.csv",'r')        #读入训练数据；open函数里的路径根据数据存储的路径来设定
training_data_list = training_data_file.readlines()
training_data_file.close()


epochs = 5                                                               #加入epocs,设定网络的训练循环次数
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')                                   #把数据依靠','区分，并分别读入
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        targets = numpy.zeros(output_nodes) + 0.01                       #设置图片与数值的对应关系
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
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01      #预处理数字图片
    outputs = n.query(inputs)                                            #让网络判断图片对应的数字
    label = numpy.argmax(outputs)                                        #找到数值最大的神经元对应的编号
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)


scores_array = numpy.asarray(scores)                                #计算图片判断的成功率
print("perfermance = ", scores_array.sum() / scores_array.size)
