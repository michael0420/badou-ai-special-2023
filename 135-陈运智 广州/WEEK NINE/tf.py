import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt

# 随机生成200个随机数字
# 生成-0.5到0.5的随机数200个，[:,np.newaxis]将上一步生成一维数组转换为列向量
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
# 生成一个服从正态分布（高斯分布）的噪声数组，均值为 0，标准差为 0.02。这个噪声数组的形状与 x_data 相同，用于模拟实际数据中的随机误差
noise =np.random.normal(0,0.02,x_data.shape)
# 将理论值与噪声相加，得到最终的带有噪声的 y 值
y_data = np.square(x_data)+noise

# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义网络中间层
Weights_l1=tf.Variable(tf.random_normal([1,10]))
biases_l1=tf.Variable(tf.zeros([1,10]))
Wx_plus_b_l1=tf.matmul(x,Weights_l1)+ biases_l1
L1 = tf.nn.tanh(Wx_plus_b_l1)

# 定义神经网络输出层
Weights_l2 =tf.Variable(tf.random_normal([10,1]))
biases_l2 =tf.Variable(tf.zeros([1,1]))
Wx_plus_b_l2= tf.matmul(L1,Weights_l2)+biases_l2
prediction=tf.nn.tanh(Wx_plus_b_l2)

#损失函数
loss =tf.reduce_mean(tf.square(y-prediction),name='loss')
# 设置学习率0.5
train_step =tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        # 训练两千次
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    # 推理
    prediction_values=sess.run(prediction,feed_dict={x:x_data})
    # 模型评估
    loss_value = sess.run(loss, feed_dict={x: x_data, y: y_data})
    print(f"Final Loss: {loss_value}")

    # 可视化
    plt.figure()
    plt.scatter(x_data, y_data, label='Original Data')
    plt.plot(x_data, prediction_values, 'r-', lw=5, label='Fitted Curve')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Neural Network Regression')
    plt.legend()
    plt.show()
