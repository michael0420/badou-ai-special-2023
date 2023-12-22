import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt
#生成随机点数据
x_data=np.linspace(-0.5,0.5,300)[:,np.newaxis]
noise=np.random.normal(0,0.03,x_data.shape)
y_data=np.square(x_data)+noise

#定义两个placeholder
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#定义隐藏层
weights_L1=tf.Variable(tf.random_normal([1,10]))
biases_l1=tf.Variable(tf.zeros([1,10]))
# print(tf.zeros([1,10]))
Wx_plus_b_L1=tf.matmul(x,weights_L1)+biases_l1
L1=tf.nn.tanh(Wx_plus_b_L1)

#输出层
weights_L2=tf.Variable(tf.random_normal([10,1]))
# print(tf.random_normal([10,1]))
biases_l2=tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2=tf.matmul(L1,weights_L2)+biases_l2
prediction=tf.nn.tanh(Wx_plus_b_L2)

#Mse
loss=tf.reduce_mean(tf.square(y-prediction))
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train,feed_dict={x:x_data,y:y_data})

    prediction_value=sess.run(prediction,feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,"r-",lw=3)
    plt.show()
