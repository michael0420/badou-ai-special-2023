# MobileNet

import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


# 卷积模块(卷积 + BN + relu6)
def conv_block(input_feature, in_channel, kernel=(3, 3), strides=(1, 1)):
    output_conv = Conv2D(in_channel,
                         kernel,
                         padding='same',
                         use_bias=False,
                         strides=strides)(input_feature)
    output = Activation(relu6)(BatchNormalization()(output_conv))

    return output


# 深度可分离卷积(深度卷积 + BN + relu6 + 卷积1x1(减少通道) + BN)
def depthwise_conv(input_feature, in_channel, depth_multiplier=1, strides=(1, 1)):
    output_depth_conv = DepthwiseConv2D((3, 3),
                                        padding='same',
                                        depth_multiplier=depth_multiplier,
                                        strides=strides,
                                        use_bias=False)(input_feature)
    output_bn_relu6 = Activation(relu6)(BatchNormalization()(output_depth_conv))

    output_conv = Conv2D(in_channel,
                         (1, 1),
                         padding='same',
                         use_bias=False,
                         strides=strides)(output_bn_relu6)
    output = Activation(relu6)(BatchNormalization()(output_conv))

    return output


def relu6(input_feature):
    output = K.relu(input_feature, max_value=6)

    return output


def mobile_net(input_shape=[224, 224, 3], depth_multiplier=1, dropout=0.001, classes=1000):
    img_input = Input(shape=input_shape)

    # (224, 224, 3) -> (112, 112, 32) -> (112, 112, 64)
    x_conv1 = conv_block(img_input, 32, strides=(2, 2))
    x_depth_conv1 = depthwise_conv(x_conv1, 64, depth_multiplier)

    # (112, 112, 64) -> (56, 56, 128) -> (56, 56, 128)
    x_conv2 = depthwise_conv(x_depth_conv1, 128, strides=(2, 2))
    x_depth_conv2 = depthwise_conv(x_conv2, 128, depth_multiplier)

    # (56, 56, 128) -> (28, 28, 256) -> (28, 28, 256)
    x_conv3 = depthwise_conv(x_depth_conv2, 256, strides=(2, 2))
    x_depth_conv3 = depthwise_conv(x_conv3, 256, depth_multiplier)

    # (28, 28, 256) -> (14, 14, 512)
    x_conv4 = depthwise_conv(x_depth_conv3, 512, strides=(2, 2))

    # 5 * [ (14, 14, 512) -> (14, 14, 512) ]
    x_depth_conv5 = depthwise_conv(x_conv4, 512, depth_multiplier)
    x_depth_conv6 = depthwise_conv(x_depth_conv5, 512, depth_multiplier)
    x_depth_conv7 = depthwise_conv(x_depth_conv6, 512, depth_multiplier)
    x_depth_conv8 = depthwise_conv(x_depth_conv7, 512, depth_multiplier)
    x_depth_conv9 = depthwise_conv(x_depth_conv8, 512, depth_multiplier)

    # (14, 14, 512) -> (7, 7, 1024) -> (7, 7, 1024)
    x_depth_conv10 = depthwise_conv(x_depth_conv9, 1024, strides=(2, 2))
    x_depth_conv11 = depthwise_conv(x_depth_conv10, 1024, depth_multiplier)

    # (7, 7, 1024) -> (1, 1, 1024)
    x_pool = GlobalAveragePooling2D()(x_depth_conv11)
    x_reshape = Reshape((1, 1, 1024))(x_pool)
    x_drop = Dropout(dropout)(x_reshape)

    # 卷积（FC替代） + softmax + reshape
    x_conv5 = Conv2D(classes, (1, 1), padding='same')(x_drop)
    output = Reshape((classes,))(Activation('softmax')(x_conv5))

    model = Model(img_input, output)
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model


# 图片像素归一化到(-1, 1)
def preprocess_input(input_feature):
    output = (input_feature / 255 - 0.5) * 2

    return output


if __name__ == '__main__':

    model = mobile_net(input_shape=[224, 224, 3])

    # 读取图片并整形(224, 224, 3)
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    # 预测并找到预测概率最大的类别作为结果
    predict = model.predict(x)
    print(np.argmax(predict))
    print('Predicted:', decode_predictions(predict, 1))

