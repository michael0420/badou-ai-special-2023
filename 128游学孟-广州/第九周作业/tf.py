import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#生成数据
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]# 创建输入数据x,范围在-0.5到0.5之间
noise = np.random.normal(0,0.02,x_data.shape) # 添加噪声
# print("x_data:",type(x_data))
# print("noise",type(noise))
y_data = np.power(x_data,4)+np.square(x_data)+noise # 生成输出数据y=x^4+x^2+noise

# 定义占位符(定义输入的形状shape和数据类型dtype)
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
# print("x",x)
# print("y",y)

# 定义神经网络结构
Weights_L1 = tf.Variable(tf.random_normal([1,10])) # 第一层权重
biases_L1 = tf.Variable(tf.zeros([1,10])) # 第一层偏置项
Wx_plus_b_L1 = tf.matmul(x,Weights_L1)+biases_L1 # 第一层输出
L1 = tf.nn.tanh(Wx_plus_b_L1) # 第一层激活函数(tanh)

Weights_L2 = tf.Variable(tf.random_normal([10,1])) #第二层权重
biases_L2 = tf.Variable(tf.zeros([1,1])) # 第二层偏置项
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2)+biases_L2 # 第二层输出
prediction = tf.nn.tanh(Wx_plus_b_L2) #加入激活函数

# 定义损失函数和优化算法
loss = tf.reduce_mean(tf.square(y-prediction))
# 定义反向传播算法（使用梯度算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练200次
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})

    #画图
    plt.figure()
    plt.scatter(x_data,y_data) #散点是真实值
    plt.plot(x_data,prediction_value,'r-',lw=5) #曲线是预测值
    plt.show()