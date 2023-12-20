import numpy
import scipy.special

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        #初始化矩阵
        #wih= 均值是0，标准差是hnodes的-0.5次方，shape是hnodes，indoes，二维矩阵
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        #sigmoid 激活函数
        self.activation_function = lambda x:scipy.special.expit(x)

        pass
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, nmdin=2).T  #转换为二维数组
        targets = numpy.array(targets_list, ndmin=2).T