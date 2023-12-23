#!/usr/bin/env pthon
# encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0.0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])
# 网络_1
w_1 = tf.Variable(tf.random.normal([1, 10]))
b_1 = tf.Variable(tf.random.normal([1, 10]))
linear_1 = tf.matmul(x, w_1) + b_1
output_1 = tf.nn.tanh(linear_1)

# 网络_2
w_2 = tf.Variable(tf.random.normal([10, 1]))
b_2 = tf.Variable(tf.random.normal([1, 1]))
linear_2 = tf.matmul(output_1, w_2) + b_2
output_2 = tf.nn.tanh(linear_2)

loss = tf.reduce_mean(tf.square(y - output_2))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
initial = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initial)
    for i in range(2000):
        sess.run(train, feed_dict={x: x_data, y: y_data})
    x_data_1 = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    noise_1 = np.random.normal(0.0, 0.02, x_data.shape)
    y_data_1 = np.square(x_data_1) + noise

    predict = sess.run(output_2, feed_dict={x: x_data_1})

plt.figure()
plt.scatter(x_data_1, y_data_1, c='green', marker='x')
plt.plot(x_data_1, predict, 'red', lw=5)
plt.show()
