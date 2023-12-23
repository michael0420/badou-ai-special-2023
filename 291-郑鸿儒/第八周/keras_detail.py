#!/usr/bin/env python
# encoding=utf-8
import numpy as np
import scipy.special


class KerasDetail:
    def __init__(self, inodes, hnodes, onodes):
        self.inodes = inodes  # 输入节点数
        self.hnodes = hnodes  # 中间节点数
        self.onodes = onodes  # 输入节点数0

        # 生成符合高斯分布的权重矩阵
        self.wih = np.random.normal(0.0, pow(hnodes, -0.5), (hnodes, inodes))
        self.who = np.random.normal(0.0, pow(onodes, -0.5), (onodes, hnodes))

        # 激活函数 sigmoid
        self.activation = lambda x: scipy.special.expit(x)

    def train(self, train_data, train_labels, lr):
        inputs = np.array(train_data, ndmin=2).T
        targets = np.array(train_labels, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        out_errors = targets - final_outputs
        # 权重使用矩阵乘，同一层之间点乘
        hidden_errors = np.dot(self.who.T, out_errors*final_outputs*(1-final_outputs))
        print(hidden_errors.shape)
        self.who += lr * np.dot(out_errors*final_outputs*(1-final_outputs), np.transpose(hidden_outputs))
        self.wih += lr * np.dot(hidden_errors*hidden_outputs*(1-hidden_outputs), np.transpose(inputs))

    def predict(self, test_data):
        inputs = np.array(test_data, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        return final_outputs


inodes = 784
hnodes = 200
onodes = 10
epochs = 5
learning_rate = 0.1

with open("data/mnist_train.csv") as train_data:
    train_data_list = train_data.readlines()

network = KerasDetail(inodes, hnodes, onodes)

for e in range(epochs):
    for line in train_data_list:
        all_values = line.split(',')
        train_data = np.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
        train_labels = np.zeros(onodes) + 0.01
        train_labels[int(all_values[0])] = 0.99
        network.train(train_data, train_labels, learning_rate)

with open("data/mnist_test.csv") as test_data:
    test_data_list = test_data.readlines()

res_arr = []
for line in test_data_list:
    all_values = line.split(',')
    test_data = np.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
    res = network.predict(test_data)
    print("网络认为当前数字是", np.argmax(res))
    print("正确数字是", all_values[0])

    if np.argmax(res) == int(all_values[0]):
        res_arr.append(1)
    else:
        res_arr.append(0)

print("正确率", np.sum(res_arr) / len(res_arr))
