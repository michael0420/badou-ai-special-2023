import numpy
import scipy.special


class NeuralNetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        #输入层
        self.inodes =input_nodes
        # 隐藏层
        self.hnodes =hidden_nodes
        # 输出层
        self.onodes =output_nodes
        # 学习率
        self.lr=learning_rate
        # 初始化权重，开始时候权重值是随机的
        
        self.Wih = (numpy.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes,self.inodes) )  )
        self.Who = (numpy.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes,self.hnodes) )  )

        # 激活函数
        # self.activation_function = lambda x: scipy.special.softmax(x, axis=0)
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    # 推理过程
    def train(self,inputs_list,targets_list):
        inputs =numpy.array(inputs_list,ndmin=2).T
        targets =numpy.array(targets_list,ndmin=2).T

        hidden_inputs =numpy.dot(self.Wih,inputs)
        hidden_outputs =self.activation_function(hidden_inputs)
        final_inputs =numpy.dot(self.Who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)

        # 计算误差
        output_errors= targets -final_outputs
        hidden_errors =numpy.dot(self.Who.T,output_errors*final_outputs*(1-final_outputs))
        self.Who+=self.lr *numpy.dot((output_errors*final_outputs*(1-final_outputs)),numpy.transpose(hidden_outputs))

        self.Wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),numpy.transpose(inputs))
        pass
    def query(self,inputs):
        # 输入层加权求和
        hidden_inputs =numpy.dot(self.Wih,inputs)
        # 激活函数得到隐藏层
        hidden_outputs =self.activation_function(hidden_inputs)
        # 隐藏层加权求和
        final_intputs =numpy.dot(self.Who,hidden_outputs)
        # 激活函数得到输出层
        final_outputs =self.activation_function(final_intputs)
        print(final_outputs)
        return final_outputs





# 初始化网络
input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.3
n =NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)
training_data_file=open('dataset/mnist_train.csv')
training_data_list=training_data_file.readlines()
training_data_file.close()
epochs =10
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]))/255.0*0.99+0.01
        targets =numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)

test_data_file=open('dataset/mnist_test.csv')
test_data_list=test_data_file.readlines()
test_data_file.close()
scores =[]
for record in test_data_list:
    all_values =record.split(',')
    correct_number =int(all_values[0])
    print("该图片对应的数字是：",correct_number)
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    outputs=n.query(inputs)
    label =numpy.argmax(outputs)
    print('输出结果是：',label)
    if label ==correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
scores_array=numpy.asarray(scores)
print('perfermance=',scores_array.sum()/scores_array.size)

