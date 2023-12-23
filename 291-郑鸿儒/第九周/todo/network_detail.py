#!/usr/bin/env python
# encoding=utf-8
import numpy as np


# 失败作，逻辑越写越复杂，再学一段时间吧。。。
class Layer(object):
    def __init__(self):
        self.input_nodes = None
        self.out_nodes = None
        self.activation = None

    def dense(self, input_nodes, out_nodes, activation):
        self.input_nodes = input_nodes
        self.out_nodes = out_nodes
        # self.activation = activation
        wih = np.random.normal(0.0, pow(out_nodes, -0.5), (out_nodes, input_nodes))
        bias = np.random.normal(size=(out_nodes,), loc=0.0, scale=0.5)
        return wih, bias, activation


class NetDetail(object):
    def __init__(self):
        self.weight_arr = []
        self.bias_arr = []
        self.activation_arr = []
        self.hidden_layer = []

    def add(self, layer_dense):
        wih, bias, activation = layer_dense
        self.weight_arr.append(wih)
        self.bias_arr.append(bias)
        self.activation_arr.append(activation)

    def fit(self, x, y, epochs, batch_size, lr):
        inputs = np.array(x, ndmin=2)
        targets = np.array(y, ndmin=2)
        batch_num = len(x) // batch_size
        for epoch in range(epochs):
            for batch in range(batch_num):
                for i in range(batch_size):
                    cur_input = inputs[batch*batch_size + i]
                    cur_target = targets[batch*batch_size + i]
                    # print(cur_input.shape)
                    hidden_input = None
                    hidden_output = None
                    self.hidden_layer = []
                    cur_hidden_in = cur_input
                    for w, b, a in zip(self.weight_arr, self.bias_arr, self.activation_arr):
                        hidden_input = np.dot(w, cur_hidden_in)
                        hidden_output = a(hidden_input)
                        self.hidden_layer.append([hidden_input, hidden_output])
                        cur_hidden_in = hidden_output
                    out_errors = cur_target - hidden_output
                    j = 0
                    derivative = []
                    for w_j, layer in zip(self.weight_arr[::-1], self.hidden_layer[::-1]):
                        hidden_input, hidden_output = layer
                        if len(derivative):
                            derivative.append(np.dot(w_j, derivative[-1] * hidden_output * (1 - hidden_output)))
                            if len(derivative) >= 2 and i < len(self.hidden_layer) - 1:
                                last_in, last_out = self.hidden_layer[::-1][j + 1]
                                derivative[-2] = np.dot(derivative[-2], last_out)
                        else:
                            derivative.append(out_errors*hidden_output * (1 - hidden_output))
                        j += 1
                    for idx, w_i in enumerate(self.weight_arr[::-1]):
                        ah = cur_input if idx > len(self.weight_arr) - 1 else\
                            self.hidden_layer[len(self.weight_arr) - i - 2]
                        self.weight_arr[len(self.weight_arr) - i - 1] += lr * derivative[idx] * ah

    def predict(self, x):
        res_arr = []
        for x_i in x:
            res = x_i
            for w, b, a in zip(self.weight_arr, self.bias_arr, self.activation_arr):
                hidden_input = np.dot(w, res)
                res = a(hidden_input)
            res_arr.append(res)
        return np.array(res_arr)


from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import net_utils as ns


(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = train_data.reshape((60000, 28*28))
train_data = train_data.astype("float32") / 255
test_data = test_data.reshape((10000, 28*28))
test_data = test_data.astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

layer_1 = Layer()
# print(layer_1.dense(10, 5, ns.sigmoid))
network = NetDetail()
network.add(layer_1.dense(28*28, 512, ns.sigmoid))
network.add(layer_1.dense(512, 256, ns.sigmoid))
network.add(layer_1.dense(256, 10, ns.sigmoid))
network.fit(train_data, train_labels, 5, 128, 0.1)
result = network.predict(test_data)
print(result)
