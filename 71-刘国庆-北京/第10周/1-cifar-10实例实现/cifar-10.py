# 该文件的目的是构造神经网络的整体结构，并进行训练和测试（评估）过程
# 导入 TensorFlow 库，将其命名为 tf
import tensorflow as tf
# 导入 NumPy 库，将其命名为 np
import numpy as np
# 导入 Python 的 time 模块
import time
# 导入 Python 的 math 模块
import math
# 导入名为 Cifar10_data 的模块
import Cifar10_data

# 设置训练的最大步数为 4000
max_steps = 4000
# 设置每个训练批次的样本数量为 100
batch_size = 100
# 用于评估的样本数量为 10000
num_examples_for_eval = 10000
# 数据集所在的目录路径为 "Cifar_data/cifar-10-batches-bin"
data_dir = "Cifar_data/cifar-10-batches-bin"


# 创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
# 定义一个带有权重损失的变量
def variable_with_weight_loss(shape, stddev, w1):
    # 创建一个形状为 shape，标准差为 stddev 的截断正态分布的变量
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    # 如果给定了权重损失参数 w1，则计算 L2 正则化损失并添加到损失集合中
    if w1 is not None:
        # 计算权重的 L2 正则化损失
        weight_loss = tf.matmul(tf.nn.l2_loss(var), w1, name='weight_loss')
        # 将权重损失添加到损失集合中
        tf.add_to_collection('losses', weight_loss)
    # 返回创建的变量
    return var


# 使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据从文件.
# 其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
# 使用 Cifar10_data 模块中的 inputs 函数加载训练集数据
# data_dir: 数据集所在的目录路径
# batch_size: 每个批次的样本数量
# distorted: 是否对训练数据进行图像失真处理
images_train, labels_train = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
# 使用 Cifar10_data 模块中的 inputs 函数加载测试集数据
# data_dir: 数据集所在的目录路径
# batch_size: 每个批次的样本数量
# distorted: 测试数据不进行图像失真处理，设置为 None
images_test, labels_test = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

# 创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。
# 要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
# 创建一个占位符用于输入图像数据，数据类型为 float32
# 形状为 [batch_size, 24, 24, 3]，表示每个批次包含 batch_size 个图像，每个图像的尺寸为 24x24 像素，通道数为 3
x = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])
# 创建一个占位符用于输入标签数据，数据类型为 int32
# 形状为 [batch_size]，表示每个批次包含 batch_size 个标签
y_ = tf.placeholder(tf.int32, shape=[batch_size])

# 创建第一个卷积层的权重，形状为 [5, 5, 3, 64]
# 表示卷积核的尺寸为 5x5，输入通道数为 3，输出通道数为 64
# 初始权重采用截断正态分布，标准差为 5e-2，权重损失参数 w1 设置为 0.0
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
# 进行卷积操作，使用步长为 1，padding 设置为 "SAME" 表示使用零填充
conv1 = tf.nn.conv2d(x, kernel1, strides=[1, 1, 1, 1], padding='SAME')
# 创建第一个卷积层的偏置，初始化为常数 0.0，形状为 [64]
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
# 对卷积结果添加偏置，并应用 ReLU 激活函数
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
# 使用最大池化进行下采样，池化窗口大小为 3x3，步长为 2，padding 设置为 "SAME"
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 创建第二个卷积层的权重，形状为 [5, 5, 64, 64]
# 表示卷积核的尺寸为 5x5，输入通道数为 64，输出通道数为 64
# 初始权重采用截断正态分布，标准差为 5e-2，权重损失参数 w1 设置为 0.0
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
# 进行卷积操作，使用步长为 1，padding 设置为 "SAME" 表示使用零填充
conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
# 创建第二个卷积层的偏置，初始化为常数 0.1，形状为 [64]
bias2 = tf.Variable(tf.constant(0.0, shape=[64]))
# 对卷积结果添加偏置，并应用 ReLU 激活函数
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
# 使用最大池化进行下采样，池化窗口大小为 3x3，步长为 2，padding 设置为 "SAME"
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
# 对第二个池化层的输出进行形状重塑，将其拉直为一维结构,-1 代表将pool2的三维结构拉直为一维结构
reshape = tf.reshape(pool2, [batch_size, -1])
# 获取 reshape 之后的第二个维度的值，即特征的维度,get_shape()[1].value 表示获取 reshape 之后的第二个维度的值
dim = reshape.get_shape()[1].value

# 建立第一个全连接层的权重，形状为 [dim, 384]
# 表示输入维度为 dim（reshape之后的特征维度），输出维度为 384
# 初始权重采用截断正态分布，标准差为 0.04，权重损失参数 w1 设置为 0.004
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
# 建立第一个全连接层的偏置，初始化为常数 0.1，形状为 [384]
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
# 对输入进行矩阵乘法运算，然后添加偏置，并应用 ReLU 激活函数
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 建立第二个全连接层的权重，形状为 [384, 192]
# 表示输入维度为 384（第一个全连接层的输出维度），输出维度为 192
# 初始权重采用截断正态分布，标准差为 0.04，权重损失参数 w1 设置为 0.004
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
# 建立第二个全连接层的偏置，初始化为常数 0.1，形状为 [192]
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
# 对第一个全连接层的输出进行矩阵乘法运算，然后添加偏置，并应用 ReLU 激活函数
fc_2 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 建立第三个全连接层的权重，形状为 [192, 10]
# 表示输入维度为 192（第二个全连接层的输出维度），输出维度为 10（对应分类的类别数）
# 初始权重采用截断正态分布，标准差为 1 / 192.0，权重损失参数 w1 设置为 0.0
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
# 建立第三个全连接层的偏置，初始化为常数 0.1，形状为 [10]
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
# 对第二个全连接层的输出进行矩阵乘法运算，然后添加偏置
result = tf.add(tf.matmul(fc_2, weight3), fc_bias3)

# 计算交叉熵损失，使用稠密标签的稀疏 softmax 交叉熵
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))
# 获取所有权重的 L2 正则化损失，并求和
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
# 总损失为交叉熵损失和权重 L2 正则化损失的和
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
# 使用 Adam 优化器进行训练，学习率为 1e-3
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
# 计算模型在 Top-K 准确率中的表现，这里 K 设置为 1
top_k_op = tf.nn.in_top_k(result, y_, 1)
# 初始化所有全局变量的操作
init_op = tf.global_variables_initializer()

# 创建 TensorFlow 会话
with tf.Session() as sess:
    # 运行全局变量初始化操作
    sess.run(init_op)
    # 启动线程操作，用于异步加载数据到队列
    tf.train.start_queue_runners()

    # 每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    # 在 TensorFlow 会话中执行训练步骤
    for step in range(max_steps):
        # 记录当前时间
        start_time = time.time()
        # 从数据集中获取一个批次的图像和标签
        image_batch, label_batch = sess.run([images_train, labels_train])
        # 运行训练操作和损失计算，同时传入图像和标签数据
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
        # 计算训练时间
        duration = time.time() - start_time
        # 每训练100步，打印损失值和训练速度信息
        if step % 100 == 0:
            # 计算每秒处理的样本数
            examples_per_sec = batch_size / duration
            # 计算每个批次的处理时间
            sec_per_batch = float(duration)
            # 打印训练信息，包括步数、损失值、每秒处理的样本数、每个批次的处理时间
            print("step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)" % (
                step, loss_value, examples_per_sec, sec_per_batch))

    # 计算最终的正确率
    # 计算评估时需要迭代的批次数量，使用 math.ceil() 向上取整
    num_batch = int(math.ceil(num_examples_for_eval / batch_size))
    # 初始化正确分类的样本计数和总样本数
    true_count = 0
    total_sample_count = num_batch * batch_size
    # 在一个for循环里面统计所有预测正确的样例个数
    # 遍历评估的批次数
    for j in range(num_batch):
        # 从测试数据集中获取一个批次的图像和标签
        image_batch, label_batch = sess.run([images_test, labels_test])
        # 运行 Top-K 准确率计算操作，同时传入图像和标签数据
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y_: label_batch})
        # 统计正确分类的样本数
        true_count += np.sum(predictions)
    # 打印正确率信息
    print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
