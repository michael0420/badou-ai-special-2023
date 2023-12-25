import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise
# plt.scatter(x_data, y_data)
# plt.show()

# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])
# print(x)

# 定义神经网络中间层
weights_l1 = tf.Variable(tf.random_normal([1, 10]))
biases_l1 = tf.Variable(tf.zeros([1, 10]))
wx_plus_b_l1 = tf.matmul(x, weights_l1) + biases_l1
output_l1 = tf.nn.tanh(wx_plus_b_l1)

# 定义神经网络输出层
weights_l2 = tf.Variable(tf.random_normal([10, 1]))
biases_l2 = tf.Variable(tf.random_normal([1, 1]))
wx_plus_b_l2 = tf.matmul(output_l1, weights_l2) + biases_l2
output_l2 = tf.nn.tanh(wx_plus_b_l2)

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - output_l2))
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    import tqdm
    pbar = tqdm.trange(2000)
    for i in pbar:
        sess.run(train_step, feed_dict={x:x_data, y:y_data})

    # 获得预测值
    prediction_value = sess.run(output_l2, feed_dict={x:x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r', lw=5)
    plt.show()













