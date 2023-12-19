import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data


epochs = 100
batch_size = 100
test_pictures = 100
data_dir = "Cifar_data/cifar-10-batches-bin"


# 设置带损失函数的矩阵
def create_matrix_with_loss(input_shape, std, weight):
    var = tf.Variable(tf.truncated_normal(input_shape, stddev=std))
    if weight:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), weight, name="weights_loss")
        tf.add_to_collection("losses", weights_loss)
    return var


# 读取数据
images_train, labels_train = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
images_test, labels_test = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)


in_shape = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [batch_size])


# 创建第一个卷积层
# input_shape=[H, W, C_in, C_out]
kernel1 = create_matrix_with_loss(input_shape=[5, 5, 3, 64], std=5e-2, weight=0.0)
conv1 = tf.nn.conv2d(in_shape, kernel1, [1, 1, 1, 1], padding="SAME")
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
# ksize, strides = [batch, height, width, channels]
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")


# 创建第二个卷积层
kernel2 = create_matrix_with_loss(input_shape=[5, 5, 64, 64], std=5e-2, weight=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# 转化为一维向量
flatten_dim = tf.reshape(pool2, [batch_size, -1])
one_dim = flatten_dim.get_shape()[1].value

# 建立第一个全连接层
weight1 = create_matrix_with_loss(input_shape=[one_dim, 384], std=0.04, weight=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(flatten_dim, weight1) + fc_bias1)

# 建立第二个全连接层
weight2 = create_matrix_with_loss(input_shape=[384, 96], std=0.04, weight=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[96]))
fc_2 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 建立第三个全连接层(方差需要归一化到0-1之间，因为最后求的是10和类别的概率值)
weight3 = create_matrix_with_loss(input_shape=[96, 10], std=1 / 96.0, weight=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.add(tf.matmul(fc_2, weight3), fc_bias3)

# 计算损失，包括权重参数的正则化损失和交叉熵损失(10个分类的损失)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))
# 获取每一条损失并加到一起(所有神经网络的损失)
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
# 总损失 = 10分类损失的平均值 + 每一个矩阵变换的损失(教学用，一般不用这个矩阵变换的损失)
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 结果中概率最高的就是预测类别
top_k_op = tf.nn.in_top_k(result, y_, 1)


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()      

# 每隔100个epoch计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for epoch in range(epochs):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={in_shape: image_batch, y_: label_batch})
        duration = time.time() - start_time

        if epoch % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (epoch, loss_value, examples_per_sec, sec_per_batch))

# 计算最终的正确率
    num_batch = int(math.ceil(test_pictures / batch_size))
    true_count = 0
    total_sample_count = num_batch * batch_size

    # 在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={in_shape: image_batch, y_: label_batch})
        true_count += np.sum(predictions)

    # 打印正确率信息
    print("accuracy = %.3f%%" % ((true_count/total_sample_count) * 100))
