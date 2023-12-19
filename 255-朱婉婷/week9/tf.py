"""
使用tf实现简单神经网络
"""

"""import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#转换成列向量
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = x_data**3 + noise

x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,1])

#定义网络中间层
w1 = tf.Variable(tf.random_normal([1,10]))
b1 = tf.Variable(tf.zeros([1,10]))
wx_b1 = tf.matmul(x,w1)+b1
l1 = tf.nn.tanh(wx_b1)

w2 = tf.Variable(tf.random_normal([10,1]))
b2 = tf.Variable(tf.zeros([1,1]))
wx_b2 = tf.matmul(l1, w2) + b2
prediction = tf.nn.tanh(wx_b2)

#定义损失函数
loss = tf.reduce_mean(tf.square(y-prediction))
#定义反向传播算法（梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
 
with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    #训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})

    #获得预测值
    prediction_value = sess.run(prediction, feed_dict={x:x_data})


    #绘制图像
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value,'r-',lw=5)
    plt.show()
"""
import  numpy as np

a = [1,1,2,3,4]
b = np.arange(4)[:,np.newaxis]
c = { "A":1,"B":2}
#print(a.items())
print(b.item())
print(c.items())