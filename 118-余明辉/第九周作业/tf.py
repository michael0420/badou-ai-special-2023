import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#自己随机数据，y = x^4+x^2

x_train = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_train.shape)
y_train = x_train**4+x_train**2+noise
y_exc = x_train**4+x_train**2


#定义输入变量
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

'''
两个隐藏层的网络,但最后结果不是很好,可能过拟合
'''
#正向推理
#定义第一个隐藏层
wh1 = tf.Variable(tf.random_uniform([1,10]))
bh1 = tf.Variable(tf.random_uniform([1,10]))
wxplusb1 = tf.matmul(x,wh1) + bh1
hout1 = tf.nn.tanh(wxplusb1)
'''
#定义第二个隐藏层
wh2 = tf.Variable(tf.random_uniform([10,10]))
bh2 = tf.Variable(tf.random_uniform([1,10]))
wxplusb2 = tf.matmul(hout1,wh2) + bh2
hout2 = tf.nn.tanh(wxplusb2)
'''

#定义输出层
wo = tf.Variable(tf.random_uniform([10,1]))
bo = tf.Variable(tf.random_uniform([1,1]))
wxplusbo = tf.matmul(hout1,wo) + bo
out = tf.nn.tanh(wxplusbo)
#反向训练
#定义损失函数
loss = tf.reduce_mean(tf.square(y-out))
#反向传播，更新权值
train = tf.train.GradientDescentOptimizer(0.1).minimize((loss))

with tf.Session() as sess:
    #定义变量之后要初始化
    sess.run(tf.global_variables_initializer())
    #训练1000次效果不好，通过调整范围3000次的拟合效果很好
    for i in range(3000):
        sess.run(train,feed_dict={x:x_train,y:y_train})
    #获得预测值
    pre = sess.run(out,feed_dict={x:x_train})
    #画图
    plt.figure()
    plt.scatter(x_train,y_train)
    plt.plot(x_train,pre,'r-')
    plt.show()


