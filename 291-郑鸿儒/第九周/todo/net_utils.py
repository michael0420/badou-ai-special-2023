#!/usr/bin/env python
# encoding=utf-8
import numpy as np


# 激活函数 activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


sigmoid.derivative = sigmoid_derivative


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def softmax(x):
    sum_exp = np.sum([np.exp(i) for i in x])
    return np.array([np.exp(i) / sum_exp for i in x])


def relu(x):
    return max(0, x)


def leaky_relu(x):
    return max(0.01 * x, x)


# 损失函数loss
def mse(y_pred, y_t):
    y_pred = np.array(y_pred)
    y_t = np.array(y_t)
    assert y_pred.shape == y_t.shape, "形状不一致"
    square_diff = np.square(y_t - y_pred)
    return np.mean(square_diff)


def crossentropy(y_pred, y_t):
    epslion = 1e-10
    y_pred = np.array(y_pred)
    y_t = np.array(y_t)
    y_pred = np.clip(y_pred, epslion, 1.0 - epslion)
    # print(y_t, y_pred.shape)

    return -np.sum(y_t*np.log(y_pred))


def crossentropy_derivative(y_pred, y_t):
    return -y_t / y_pred


crossentropy.derivative = crossentropy_derivative
