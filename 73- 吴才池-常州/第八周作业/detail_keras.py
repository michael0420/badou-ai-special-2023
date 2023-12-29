import numpy
import scipy.special

class NeuralNetWork:
    def __init__(self,inputnodes,hidennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hidennodes
        self.onodes=outputnodes

        self.lr=learningrate

        #  初始化随机权值
        self.wih=(numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
        self.who=(numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes)))

        # 定义激活函数sigmoid
        self.activation_function=lambda x:scipy.special.expit(x)

        pass

    def train(self,inputs_list,targets_list):
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T
        #输入到隐藏
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        #隐藏到输出
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)

        #计算误差
        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.who.T,output_errors*final_outputs*(1-final_outputs))

        #更新权值
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass
    def query(self,inputs):
        hidden_inpputs=numpy.dot(self.wih,inputs)

        hidden_outputs=self.activation_function(hidden_inpputs)

        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)

        print(final_outputs)
        return final_outputs

#初始化

input_nodes=784
hidden_nodes=200
output_nodes=10
leaning_rate=0.1
n=NeuralNetWork(input_nodes,hidden_nodes,output_nodes,leaning_rate)
#读入数据集
train_data_file=open("dataset/mnist_train.csv",'r')
train_data_list=train_data_file.readlines()
train_data_file.close()

#设定训练相关参数
epochs=5
for e in range(epochs):
    for record in train_data_list:
        all_values=record.split(',')

        inputs=(numpy.asfarray(all_values[1:]))/255.0*0.99+0.01

        targets=numpy.zeros(output_nodes)+0.01
        print(targets)
        targets[int(all_values[0])]=0.99
        print(int(all_values[0]))
        n.train(inputs,targets)

test_data_file=open("dataset/mnist_test.csv")
test_data_list=test_data_file.readlines()
test_data_file.close()
scores=[]
for record in test_data_list:
    all_values=record.split(",")
    correct_number=int(all_values[0])
    print("answer:",correct_number)

    inputs=(numpy.asfarray(all_values[1:]))/255.0*0.99+0.01

    outputs=n.query(inputs)

    label=numpy.argmax(outputs)


    print("网络认为的数字是：" ,label)

    if(label==correct_number):
        scores.append(1)
    else:
        scores.append(0)

print(scores)

scores_array=numpy.asarray(scores)
print("performance=",scores_array.sum()/scores_array.size)
