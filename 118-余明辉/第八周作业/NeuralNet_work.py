import scipy.special
import numpy as np

class NeuralNet:

    def __init__(self,Nin,Nhide,Nout,learn):
        #初始化 输入，隐藏层，输出节点个数 和 学习率
        self.Nin = Nin
        self.Nhide = Nhide
        self.Nout = Nout
        self.learn = learn

        #定义激活函数 sigmoid
        self.acF = lambda x:scipy.special.expit(x)

        #随机初始化权重 wh * in = hide ; wo * hide = out
        self.wh = np.random.normal(0.0,pow(self.Nhide,-0.5),(self.Nhide,self.Nin))
        self.wo = np.random.normal(0.0,pow(self.Nout,-0.5),(self.Nout,self.Nhide))

    def train(self,input,traget):
        #将数据变成2维
        inputs = np.array(input,ndmin = 2).T
        tragets = np.array(traget,ndmin = 2).T

        # 隐藏层计算
        in_hide = np.dot(self.wh, inputs)
        out_hide = self.acF(in_hide)
        # 输出层计算
        in_out = np.dot(self.wo, out_hide)
        out_out = self.acF(in_out)

        #反向
        #误差
        outerrors = tragets - out_out
        hideerrors = np.dot(self.wo.T,outerrors*out_out*(1-out_out))
        #更新权重
        self.wo += self.learn * np.dot((outerrors * out_out * (1 - out_out)),
                                        np.transpose(out_hide))
        self.wh += self.learn * np.dot((hideerrors * out_hide * (1 - out_hide)),
                                        np.transpose(inputs))

    def query(self,input):
        #隐藏层计算
        in_hide = np.dot(self.wh,input)
        out_hide = self.acF(in_hide)
        #输出层计算
        in_out = np.dot(self.wo,out_hide)
        ans = self.acF(in_out)

        return ans

'''
初始化
'''
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

'''
读入数据
'''
trainfile = open("dataset/mnist_train.csv",'r')
training_data_list = trainfile.readlines()
trainfile.close()
'''
开始训练
'''
epoch = 5 #用数据训练五次
for i in range(epoch):
    #通过每一组数据进行训练
    for x in training_data_list:
        x = x.split(',')
        train = np.asfarray(x[1:])/255*0.99 + 0.01
        targ = np.zeros(output_nodes)
        targ[int(x[0])] = 1
        n.train(train, targ)

'''
正向 测试
'''

testfile = open("dataset/mnist_test.csv",'r')
testing_data_list = testfile.readlines()
testfile.close()
exact = [] #记录是否能准确
for x in testing_data_list:
    x = x.split(',')
    test = np.asfarray(x[1:])/255*0.99+0.01
    #找到最大数字所对应的索引值
    ans = np.argmax(n.query(test))
    print(x[0],ans)
    if ans == int(x[0]):
        exact.append(1)
    else:
        exact.append(0)

print(exact)

