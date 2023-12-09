import numpy
import scipy.special


class NeuralNetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.in_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate
        self.w_ih = numpy.random.normal(0, pow(self.h_nodes, -0.7), (self.h_nodes, self.in_nodes))
        self.w_ho = numpy.random.normal(0, pow(self.o_nodes, -0.7), (self.o_nodes, self.h_nodes))
        self.active_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        # 根据输入的训练数据更新节点链路权重
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        '''
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = numpy.dot(self.w_ih, inputs)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.active_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.active_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.w_ho.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.w_ho += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                         numpy.transpose(hidden_outputs))
        self.w_ih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                         numpy.transpose(inputs))

        pass

    def query(self, inputs):
        hidden_inputs = numpy.dot(self.w_ih, inputs)
        hidden_outputs = self.active_function(hidden_inputs)
        final_outputs = numpy.dot(self.w_ho, hidden_outputs)
        final_outputs = self.active_function(final_outputs)
        return final_outputs


if __name__ == "__main__":
    input_node2 = 784
    hidden_nodes = 200
    output_nodes = 10
    learn_rate = 0.1
    epochs = 10
    n = NeuralNetWork(input_node2, hidden_nodes, output_nodes, learn_rate)

    train_data_file = open('mnist_train.csv')
    train_data_list = train_data_file.readlines()
    train_data_file.close()
    for e in range(epochs):
        for record in train_data_list:
            all_values = record.split(',')
            inputs = numpy.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    test_data_file = open("mnist_test.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        print("数字为：", correct_number)
        inputs = numpy.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        print("网络数字判断的数字是：", label)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
        # 计算图片判断的成功率
    scores_array = numpy.asarray(scores)
    print("perfermance = ", scores_array.sum() / scores_array.size)
