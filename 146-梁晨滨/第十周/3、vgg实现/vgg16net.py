import tensorflow as tf

# 创建slim对象
slim = tf.contrib.slim


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):

    with tf.variable_scope(scope, 'vgg_16', [inputs]):

        # slim.repeat(输入，重复次数，类型，通道，卷积核，名称）slim.max_pool2d(输入，池化核，名称)
        # 2个卷积 + 最大池化 output = (112, 112, 64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # 2个卷积 + 最大池化  output = (56, 56, 128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # 3个卷积 + 最大池化  output = (28, 28, 256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # 3个卷积 + 最大池化  output = (14, 14, 512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # 3个卷积 + 最大池化  output = (14, 14, 512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 1x1卷积替代池化 output = (1,1,4096)
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')

        # 1x1卷积替代全连接层 output = (1,1,4096)
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

        # 1x1卷积替代全连接层 output = (1,1,1000)
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
        
        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')

        return net