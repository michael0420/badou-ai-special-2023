import tensorflow as tf  
import numpy as np
import matplotlib.pyplot as plt

# 输入是[200, 1]格式，每个数是-1到1之间均匀分布
# 加入均值为0，标准差为0.05的噪声
input_random = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.05, input.shape)
input = np.square(input_random) + noise


input1 = tf.placeholder(tf.float32, [None,1])
ground_truth = tf.placeholder(tf.float32, [None,1])

# 定义模型第一层(输入到隐藏层)
# y = k*x + b 和一个激活函数
weight1 = tf.Variable(tf.random_normal([1, 10]))
bias1 = tf.Variable(tf.zeros([1, 10]))
learnable_matrix1 = tf.matmul(input1, weight1) + bias1
output1 = tf.nn.relu(learnable_matrix1)

# 定义模型第二层(隐藏层到输出层)
weight2 = tf.Variable(tf.random_normal([10, 1]))
bias2 = tf.Variable(tf.zeros([1, 1]))
learnable_matrix1 = tf.matmul(output1, weight2) + bias2
output2 = tf.nn.relu(learnable_matrix1)

# 定义损失
loss = tf.reduce_mean(tf.square(ground_truth - output2))
# 定义反向传播
backward = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

if __name__ == "__main__":
    with tf.Session() as sess:
        # 训练
        sess.run(tf.global_variables_initializer())
        epoches = 1000
        for i in range(epoches):
            sess.run(backward, feed_dict={input_random: input, input: ground_truth})

        # 测试
        predict = sess.run(output2, feed_dict={input_random: input})

        # 画图
        plt.figure()
        plt.scatter(input, ground_truth)
        plt.plot(input, predict, 'r-', lw=5)
        plt.show()


