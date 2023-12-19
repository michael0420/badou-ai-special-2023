import numpy as np
import scipy.special
import cv2


class OwnModel:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate,actiname ):
        # 定义输入，隐藏，输出节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 定义学习率
        self.lr = learningrate

        #定义激活函数
        if(actiname == 'relu'):
            self.activation_function = lambda x: np.maximum(0,x)
        elif(actiname == 'sigmod'):
            self.activation_function = lambda x: scipy.special.expit(x)
        else:
            self.activation_function = lambda x: scipy.special.expit(x)
        # 定义权重矩阵 ，输入层的权重矩阵
        self.in_weight = np.random.rand(self.hnodes, self.inodes) - 0.5
        # 隐藏层的权重矩阵
        self.out_weight = np.random.rand(self.onodes, self.hnodes) - 0.5
        pass

    def fit(self, inputs,epoch):
        # 总的训练集迭代多少次
        for i in range(epoch):
            input_copy = inputs
            # 便利每个训练集数据
            for record in input_copy:
                all_values = record.split(',')
                input_copy = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
                # 设置图片与数值的对应关系
                targets = np.zeros(self.onodes) + 0.01
                targets[int(all_values[0])] = 0.99
                self.onetrain(input_copy, targets)

    def onetrain(self,inputs_list,targets_list):
        # # 获取训练的数据
        inputs = np.array(inputs_list, ndmin=2).T
        # 获取训练数据的结果标签
        targets = np.array(targets_list, ndmin=2).T
        # 计算第一层：
        # 信号经过输入层后产生的信号量，即 Z 的值
        hidden_inputs = np.dot(self.in_weight, inputs)
        # #  Z 做激活函数，得到 A 的值
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算第二层
        # 输出层接收来自中间隐藏层的信号量 ，即Z的值
        final_inputs = np.dot(self.out_weight, hidden_outputs)
        # 输出层的Z 做激活函数，得到A 的值
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        # # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上，
        # # 根据梯度公式 - （t - Ot）* sigmod * (1-sigmod) * Ok （上个节点的值）
        self.out_weight = self.out_weight + self.lr * np.dot((output_errors  * final_outputs * (1 - final_outputs)),
                                        np.transpose(hidden_outputs))

        # # 根据梯度公式 - （t - Ot）* sigmod * (1-sigmod) * Ok （上个节点的值）
        # # 因为hidden_error 是经过两层影响的，需要剥离出 最后隐藏层上的误差
        hidden_errors = np.dot(self.out_weight.T, output_errors * final_outputs * (1 - final_outputs))
        self.in_weight = self.in_weight + self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        np.transpose(inputs))
        pass
    def query(self, inputs):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.in_weight, inputs)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = np.dot(self.out_weight, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def evaluate(self, test_data_list):
        scores = []
        for record in test_data_list:
            all_values = record.split(',')
            correct_number = int(all_values[0])
            # 预处理数字图片
            inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
            # 让网络判断图片对应的数字,推理
            outputs = self.query(inputs)
            # 找到数值最大的神经元对应的 编号
            label = np.argmax(outputs)
            if label == correct_number:
                scores.append(1)
            else:
                scores.append(0)
        print(scores)

        # 计算图片判断的成功率
        scores_array = np.asarray(scores)
        perfermance = scores_array.sum() / scores_array.size
        return perfermance


if __name__ == '__main__':
    # 1、构建模型
    model = OwnModel(28*28,100,10,0.4,'sigmod')
    # 2、构建训练集和测试集
    training_data_file = open("dataset/mnist_train.csv")
    trainning_data_list = training_data_file.readlines()
    training_data_file.close()
    # 3、训练模型
    model.fit(trainning_data_list,20)
    # 4、评估模型
    test_data_file = open("dataset/mnist_test.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    perfermance = model.evaluate(test_data_list)
    print('perfermance:', perfermance)
    # 5、预测值
    img = cv2.imread("dataset/my_own_4.png")

    img_1 = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), (28,28), interpolation=cv2.INTER_CUBIC)
    height, width, _ = img.shape
    pixels = []
    for y in range(height):
        for x in range(width):
            # 获取像素值
            pixel = img_1[y, x]
            # 将像素值添加到集合中
            pixels.append(pixel)
    queryData = np.ravel(pixels)
    final_outputs = model.query(queryData)
    index = np.argmax(final_outputs)
    print("the number for the picture is : ", index)
    pass






