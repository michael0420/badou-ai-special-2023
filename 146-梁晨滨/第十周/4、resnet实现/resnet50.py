from __future__ import print_function

import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model

from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def res_block(input_tensor, kernel_size, channel):

    c1, c2, c3 = channel

    # 1x1卷积 + BN + relu
    conv_combine_out1 = Activation('relu')(BatchNormalization()(Conv2D(c1, (1, 1))(input_tensor)))
    # 3x3卷积 + BN + relu
    conv_combine_out2 = Activation('relu')(BatchNormalization()(Conv2D(c2, kernel_size, padding='same')(conv_combine_out1)))
    # 3x3卷积 + BN
    conv_combine_out3 = BatchNormalization()(Conv2D(c3, (1, 1))(conv_combine_out2))

    #  残差结构 relu (卷积结果 + 输入)
    out = Activation('relu')(layers.add([conv_combine_out3, input_tensor]))

    return out


def conv_block(input_tensor, kernel_size, channel, strides=(2, 2)):

    c1, c2, c3 = channel

    # 1x1卷积 + BN + relu
    conv_combine_out1 = Activation('relu')(BatchNormalization()(Conv2D(c1, (1, 1), strides=strides)(input_tensor)))

    # conv_combine_out1 = Activation('relu')(BatchNormalization(Conv2D(c1, (1, 1), strides=strides)(input_tensor)))
    # 3x3卷积 + BN + relu
    conv_combine_out2 = Activation('relu')(BatchNormalization()(Conv2D(c2, kernel_size, padding='same')(conv_combine_out1)))
    # 1x1卷积 + BN
    conv_combine_out3 = BatchNormalization()(Conv2D(c3, (1, 1))(conv_combine_out2))

    # 残差
    shortcut = BatchNormalization()(Conv2D(c3, (1, 1), strides=strides)(input_tensor))

    # relu (卷积结果 + 残差)
    out = Activation('relu')(layers.add([conv_combine_out3, shortcut]))

    return out


def resnet50(classes=1000):

    img_input = Input(shape=[224, 224, 3])
    img_input_padding = ZeroPadding2D((3, 3))(img_input)

    # 一阶段
    # 3x3conv + bn + relu + max_pooling
    conv_combine_out1 = MaxPooling2D((3, 3), strides=(2, 2))(Activation('relu')(BatchNormalization()(Conv2D(64, (7, 7), strides=(2, 2))(img_input_padding))))

    # 二阶段
    # conv_block + 2个identity_block
    x = conv_block(conv_combine_out1, 3, [64, 64, 256], strides=(1, 1))
    x = res_block(x, 3, [64, 64, 256])
    x = res_block(x, 3, [64, 64, 256])

    # 三阶段
    # conv_block + 3个identity_block
    x = conv_block(x, 3, [128, 128, 512])
    x = res_block(x, 3, [128, 128, 512])
    x = res_block(x, 3, [128, 128, 512])
    x = res_block(x, 3, [128, 128, 512])

    # 四阶段
    # conv_block + 5个identity_block
    x = conv_block(x, 3, [256, 256, 1024])
    x = res_block(x, 3, [256, 256, 1024])
    x = res_block(x, 3, [256, 256, 1024])
    x = res_block(x, 3, [256, 256, 1024])
    x = res_block(x, 3, [256, 256, 1024])
    x = res_block(x, 3, [256, 256, 1024])

    # 五阶段
    # conv_block + 2个identity_block
    x = conv_block(x, 3, [512, 512, 2048])
    x = res_block(x, 3, [512, 512, 2048])
    x = res_block(x, 3, [512, 512, 2048])

    x = AveragePooling2D((7, 7))(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(img_input, x)

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model


if __name__ == '__main__':
    model = resnet50()
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    pred = model.predict(x)
    print('Predicted:', decode_predictions(pred))
