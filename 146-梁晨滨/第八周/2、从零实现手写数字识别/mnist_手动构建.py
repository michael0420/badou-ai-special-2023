import numpy as np
import scipy.special

# 模型构建
class Model:
    def __init__(self, in_dim, middle_dim, out_dim, learningrate):
        self.in_dim = in_dim
        self.mid_dim = middle_dim
        self.out_dim = out_dim
        self.lr = learningrate

        # self.in_mid(输入到中间层的变换矩阵), self.mid_out(中间层到输出的变换矩阵)
        self.variance = pow(self.mid_dim, -0.5)
        self.in_mid = np.random.normal(0, self.variance, (self.mid_dim, self.in_dim))
        self.mid_out = np.random.normal(0, self.variance, (self.out_dim, self.mid_dim ))

        self.sig = lambda x: scipy.special.expit(x)

    # 定义训练过程
    def train(self, inputs, groud_truth):

        inputs = np.array(inputs, ndmin=2).T
        gt = np.array(groud_truth, ndmin=2).T

        # 输入到中间层，经过一层变换矩阵和sigmod激活函数
        in_to_mid = self.sig(np.dot(self.in_mid, inputs))
        # 中间层到输出层重复上一步过程
        mid_to_out = self.sig(np.dot(self.mid_out, in_to_mid))

        # 计算误差
        out_error = gt - mid_to_out
        mid_error = np.dot(self.mid_out.T, out_error * mid_to_out * (1 - mid_to_out))

        # 根据误差更新两个变换矩阵
        self.mid_out += self.lr * np.dot(out_error * mid_to_out * (1 - mid_to_out), np.transpose(in_to_mid))
        self.in_mid += self.lr * np.dot(mid_error * in_to_mid * (1 - in_to_mid), np.transpose(inputs))

    # 定义测试过程
    def test(self, inputs):

        in_to_mid = self.sig(np.dot(self.in_mid, inputs))
        mid_to_out = self.sig(np.dot(self.mid_out, in_to_mid))

        return mid_to_out


if __name__ == "__main__":
    # 设置相关参数
    in_dim, middle_dim, out_dim, lr = 28 * 28, 256, 10, 0.2
    net = Model(in_dim, middle_dim, out_dim, lr)

    """训练"""
    # 按行读取csv文件，(with open自动关闭文件， 文件简称f)
    with open("dataset/mnist_train.csv", 'r') as f:
        training_data_list = f.readlines()

    epochs = 10
    for i in range(epochs):
        for every_data in training_data_list:
            value = every_data.split(',')
            # 输入的第0维度是图像中的手写数字值，第1维度是图像矩阵
            # 输入的每个像素值归一化到0-1， 为了使大部分不为0，加一个0.01
            inputs = (np.asfarray(value[1:]))/255.0 * 0.99 + 0.01
            # 设置gt的one-hot编码，例如【0， 0， 0， 1， 0】代表图片是数字4
            # 为了不过于绝对，设置为0.99
            ground_truth = np.zeros(out_dim) + 0.01
            ground_truth[int(value[0])] = 0.99
            # 训练
            net.train(inputs, ground_truth)

    """测试"""
    with open("dataset/mnist_test.csv", 'r') as f:
        test_data_list = f.readlines()

    scores = []
    for i in test_data_list:
        value = i.split(',')
        gt_num = int(value[0])
        print("该图片对应的数字为:", gt_num)
        test_pic = (np.asfarray(value[1:])) / 255.0 * 0.99 + 0.01
        predict_list = net.test(test_pic)
        predict = np.argmax(predict_list)
        print("预测数字为:", predict)


        if predict == gt_num:
            scores.append(1)
        else:
            scores.append(0)

    print("预测结果为(1是对，0是错)：", scores)

    accuracy = sum(scores)/len(scores)
    print("测试集准确率：", accuracy)





