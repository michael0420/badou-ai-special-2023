import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]       #使一维数据转变为二维数据，要使一维数据转变为三维数据为[:,np.newaxis,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise

x = tf.placeholder(tf.float32,shape=[None,1])
y = tf.placeholder(tf.float32,shape=[None,1])

w_l1 = tf.Variable(tf.random.normal([1,10]))
b_l1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_l1=tf.matmul(x,w_l1)+b_l1
L1=tf.nn.tanh(Wx_plus_b_l1)

w_l2 = tf.Variable(tf.random.normal([10,1]))
b_l2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_l2=tf.matmul(L1,w_l2)+b_l2
result=tf.nn.tanh(Wx_plus_b_l2)

loss = tf.reduce_mean(tf.square(y-result))     #tf.reduce_mean用于计算平均值
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:                  #tf.Session()的括号不要漏了
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train,feed_dict={x:x_data,y:y_data})

#最后的预测结果在for循环外
    result = sess.run(result,feed_dict={x:x_data})

plt.figure()
plt.scatter(x_data,y_data)
plt.plot(x_data,result,'r-',lw=5)
plt.show()

