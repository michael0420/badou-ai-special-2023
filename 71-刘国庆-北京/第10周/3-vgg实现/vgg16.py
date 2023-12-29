# 导入 TensorFlow 库
import tensorflow as tf

# 创建slim对象
slim = tf.contrib.slim


# 使用 TensorFlow 变量域来定义一个 VGG-16 网络的函数
def vgg_16(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.5, spatial_squeeze=True, scope='vgg_16'):
    # 在给定的变量域内，建立 VGG-16 网络的结构
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        # 第一段卷积层（conv1）：两个[3,3]的卷积层，输出特征层为64，得到(224,224,64)的输出
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 第一段池化层（pool1）：2x2 的最大池化，输出为(112,112,64)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # 第二段卷积层（conv2）：两个[3,3]的卷积层，输出特征层为128，得到(112,112,128)的输出
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 第二段池化层（pool2）：2x2 的最大池化，输出为(56,56,128)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # 第三段卷积层（conv3）：三个[3,3]的卷积层，输出特征层为256，得到(56,56,256)的输出
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # 第三段池化层（pool3）：2x2 的最大池化，输出为(28,28,256)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # 第四段卷积层（conv4）：三个[3,3]的卷积层，输出特征层为512，得到(28,28,512)的输出
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # 第四段池化层（pool4）：2x2 的最大池化，输出为(14,14,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # 第五段卷积层（conv5）：三个[3,3]的卷积层，输出特征层为512，得到(14,14,512)的输出
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # 第五段池化层（pool5）：2x2 的最大池化，输出为(7,7,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        # 第一个全连接层（fc6）：利用卷积模拟全连接，输出特征层为4096，得到(1,1,4096)的输出
        # 定义 VGG-16 网络的第一个全连接层（fc6）：应用 7x7 大小的卷积核，输出特征层为 4096
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        # 在全连接层的输出上应用随机失活：保留节点的概率为 dropout_keep_prob，在训练模式下进行随机失活
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
        # 第二个全连接层（fc7）：利用卷积模拟全连接，输出特征层为4096，得到(1,1,4096)的输出
        # 定义 VGG-16 网络的第二个全连接层（fc7）：应用 1x1 大小的卷积核，输出特征层为 4096
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        # 在全连接层的输出上应用随机失活：保留节点的概率为 dropout_keep_prob，在训练模式下进行随机失活
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
        # 第三个全连接层（fc8）：利用卷积模拟全连接，输出特征层为num_classes（默认为1000），得到(1,1,num_classes)的输出
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
        # 由于使用卷积方式模拟全连接层，需要对输出进行降维
        # 如果 spatial_squeeze 为 True，利用 TensorFlow 的 squeeze 函数对输出进行降维
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        # 返回 VGG-16 网络的最终输出
        return net
