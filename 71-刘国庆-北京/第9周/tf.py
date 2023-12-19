import tensorflow.compat.v1 as tf  # 导入TensorFlow v1版本
import numpy as np  # 导入NumPy库
import matplotlib.pyplot as plt  # 导入Matplotlib库用于绘图
import os  # 导入操作系统相关的库

# Step 0:设置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Step 1:使用numpy生成200个随机点
# 生成一维数组x_data
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# 生成噪声
noise = np.random.normal(0, 0.02, x_data.shape)
# 生成y_data，加入噪声
y_data = np.square(x_data) + noise

# Step 2:定义两个placeholder存放输入数据
# 定义输入占位符x
x = tf.placeholder(tf.float32, [None, 1])
# 定义标签占位符y
y = tf.placeholder(tf.float32, [None, 1])

# Step 3:定义神经网络中间层
# 定义中间层权重
Weight_L1 = tf.Variable(tf.random_normal([1, 10]))
# 定义中间层偏置项
Bias_L1 = tf.Variable(tf.zeros([1, 10]))
# 中间层输入
Middle_input = tf.matmul(x, Weight_L1) + Bias_L1
# 中间层激活函数
L1 = tf.nn.tanh(Middle_input)

# Step 4:定义神经网络输出层
# 定义输出层权重
Weight_L2 = tf.Variable(tf.random_normal([10, 1]))
# 定义输出层偏置项
Bias_L2 = tf.Variable(tf.zeros([1, 1]))
# 输出层输入
Out_input = tf.matmul(L1, Weight_L2)
# 输出层激活函数
L2 = tf.nn.tanh(Out_input)

# Step 5:定义损失函数(均方差函数)
# 计算均方差损失
loss = tf.reduce_mean(tf.square(y - L2))

# Step 6:定义反向传播算法(使用梯度下降算法训练)
# 梯度下降优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# Step 7:创建TensorFlow会话
with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())
    # 运行训练步骤，传入数据
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    # 运行预测操作，传入数据
    predict = sess.run(Out_input, feed_dict={x: x_data})
    # Step 8:画图
    plt.figure()
    # 绘制散点图，表示真实值
    plt.scatter(x_data, y_data)
    # 绘制曲线图，表示预测值
    plt.plot(x_data, predict, 'r-', lw=5)
    # 显示图形
    plt.show()
