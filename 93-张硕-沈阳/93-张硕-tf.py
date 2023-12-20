import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  #生成-0.5 到 0.5之间200个等距数值，并转换为二维
noise = np.random.normal(0, 0.02, x_data.shape) #生成和x_data同样shape, 符合μ为0，标准差为0.02的数组
y_data = np.square(x_data) + noise

#存放输入数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

#定义中间层
w1 = tf.Variable(tf.random_normal([1,10]))
b1 = tf.Variable(tf.zeros([1,10]))
out1 = tf.matmul(x, w1) + b1
L1 = tf.nn.tanh(out1)

#定义输出层
w2 = tf.Variable(tf.random_normal([10,1]))
b2 = tf.Variable(tf.zeros([1,1]))
out2 = tf.matmul(L1, w2) + b2
L2 = tf.nn.tanh(out2)

#损失函数
loss = tf.reduce_mean(tf.square(y - L2))
#梯度下降
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})

    prediction_value = sess.run(L2, feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data, y_data)   #散点是真实值
    plt.plot(x_data,prediction_value,'r-',lw=5)   #曲线是预测值
    plt.show()
