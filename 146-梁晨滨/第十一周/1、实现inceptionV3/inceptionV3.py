# inceptionv3

import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import GlobalAveragePooling2D, Activation,Dense,Input,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras.preprocessing import image


# 卷积模块(卷积 + BN + relu)
def conv_block(input_feature, in_channel, kernel=(3, 3), strides=(1, 1), padding='same'):
    output_conv = Conv2D(in_channel,
                         kernel,
                         strides=strides,
                         padding=padding,
                         use_bias=False,
                         )(input_feature)
    output = Activation('relu')(BatchNormalization(scale=False)(output_conv))

    return output


def inceptionV3(input_shape=[299, 299, 3], classes=1000):
    img_input = Input(shape=input_shape)

    # (229, 229, 3) -> (149, 149, 32) -> (147, 147, 32) -> (147, 147, 64) -> (73, 73, 64)
    x_conv1 = conv_block(img_input, 32, kernel=(3, 3), strides=(2, 2), padding='valid')
    x_conv2 = conv_block(x_conv1, 32, kernel=(3, 3), padding='valid')
    x_conv3 = conv_block(x_conv2, 64, kernel=(3, 3))
    x_pool = MaxPooling2D((3, 3), strides=(2, 2))(x_conv3)

    # (73, 73, 64) -> (71, 71, 80) -> (35, 35, 192) -> (35, 35, 256)
    x_conv4 = conv_block(x_pool, 80, kernel=(1, 1), padding='valid')
    x_conv5 = conv_block(x_conv4, 192, kernel=(3, 3), padding='valid')
    x_pool2 = MaxPooling2D((3, 3), strides=(2, 2))(x_conv5)

    # 第1阶段(一共3部分)

    # 第1阶段->第1部分
    stage1_part1_branch1 = conv_block(x_pool2, 64, kernel=(1, 1))

    stage1_part1_branch2 = conv_block(x_pool2, 48, kernel=(1, 1))
    stage1_part1_branch2 = conv_block(stage1_part1_branch2, 64, kernel=(5, 5))

    stage1_part1_branch3 = conv_block(x_pool2, 64, kernel=(1, 1))
    stage1_part1_branch3 = conv_block(stage1_part1_branch3, 96, kernel=(3, 3))
    stage1_part1_branch3 = conv_block(stage1_part1_branch3, 96, kernel=(3, 3))

    stage1_part1_branch4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x_pool2)
    stage1_part1_branch4 = conv_block(stage1_part1_branch4, 32, kernel=(1, 1))

    output1 = layers.concatenate([stage1_part1_branch1, stage1_part1_branch2, stage1_part1_branch3, stage1_part1_branch4],
                                 axis=3)
    # 第1阶段->第2部分
    stage1_part2_branch1 = conv_block(output1, 64, kernel=(1, 1))

    stage1_part2_branch2 = conv_block(output1, 48, kernel=(1, 1))
    stage1_part2_branch2 = conv_block(stage1_part2_branch2, 64, kernel=(5, 5))

    stage1_part2_branch3 = conv_block(output1, 64, kernel=(1, 1))
    stage1_part2_branch3 = conv_block(stage1_part2_branch3, 96, kernel=(3, 3))
    stage1_part2_branch3 = conv_block(stage1_part2_branch3, 96, kernel=(3, 3))

    stage1_part2_branch4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(output1)
    stage1_part2_branch4 = conv_block(stage1_part2_branch4, 64, kernel=(1, 1))

    output2 = layers.concatenate([stage1_part2_branch1, stage1_part2_branch2, stage1_part2_branch3, stage1_part2_branch4],
                                 axis=3)

    # 第1阶段->第3部分
    stage1_part3_branch1 = conv_block(output2, 64, kernel=(1, 1))

    stage1_part3_branch2 = conv_block(output2, 48, kernel=(1, 1))
    stage1_part3_branch2 = conv_block(stage1_part3_branch2, 64, kernel=(5, 5))

    stage1_part3_branch3 = conv_block(output2, 64, kernel=(1, 1))
    stage1_part3_branch3 = conv_block(stage1_part3_branch3, 96, kernel=(3, 3))
    stage1_part3_branch3 = conv_block(stage1_part3_branch3, 96, kernel=(3, 3))

    stage1_part3_branch4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(output2)
    stage1_part3_branch4 = conv_block(stage1_part3_branch4, 64, kernel=(1, 1))

    output3 = layers.concatenate(
        [stage1_part3_branch1, stage1_part3_branch2, stage1_part3_branch3, stage1_part3_branch4],
        axis=3)

    # 第2阶段(一共5部分)

    # 第2阶段->第1部分
    stage2_part1_branch1 = conv_block(output3, 384, kernel=(3, 3), strides=(2, 2), padding='valid')

    stage2_part1_branch2 = conv_block(output3, 64, kernel=(1, 1))
    stage2_part1_branch2 = conv_block(stage2_part1_branch2, 96, kernel=(3, 3))
    stage2_part1_branch2 = conv_block(stage2_part1_branch2, 96, kernel=(3, 3), strides=(2, 2), padding='valid')

    stage2_part1_branch3 = MaxPooling2D((3, 3), strides=(2, 2))(output3)

    output4 = layers.concatenate(
        [stage2_part1_branch1, stage2_part1_branch2, stage2_part1_branch3], axis=3)

    # 第2阶段->第2部分
    stage2_part2_branch1 = conv_block(output4, 192, kernel=(1, 1))

    stage2_part2_branch2 = conv_block(output4, 128, kernel=(1, 1))
    stage2_part2_branch2 = conv_block(stage2_part2_branch2, 128, kernel=(1, 7))
    stage2_part2_branch2 = conv_block(stage2_part2_branch2, 192, kernel=(7, 1))

    stage2_part2_branch3 = conv_block(output4, 128, kernel=(1, 1))
    stage2_part2_branch3 = conv_block(stage2_part2_branch3, 128, kernel=(7, 1))
    stage2_part2_branch3 = conv_block(stage2_part2_branch3, 128, kernel=(1, 7))
    stage2_part2_branch3 = conv_block(stage2_part2_branch3, 128, kernel=(7, 1))
    stage2_part2_branch3 = conv_block(stage2_part2_branch3, 192, kernel=(1, 7))

    stage2_part2_branch4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(output4)
    stage2_part2_branch4 = conv_block(stage2_part2_branch4, 192, 1, 1)

    output5 = layers.concatenate(
        [stage2_part2_branch1, stage2_part2_branch2, stage2_part2_branch3, stage2_part2_branch4], axis=3)

    # 第2阶段->第3部分
    stage2_part3_branch1 = conv_block(output5, 192, kernel=(1, 1))

    stage2_part3_branch2 = conv_block(output5, 160, kernel=(1, 1))
    stage2_part3_branch2 = conv_block(stage2_part3_branch2, 160, kernel=(1, 7))
    stage2_part3_branch2 = conv_block(stage2_part3_branch2, 192, kernel=(7, 1))

    stage2_part3_branch3 = conv_block(output5, 160, kernel=(1, 1))
    stage2_part3_branch3 = conv_block(stage2_part3_branch3, 160, kernel=(7, 1))
    stage2_part3_branch3 = conv_block(stage2_part3_branch3, 160, kernel=(1, 7))
    stage2_part3_branch3 = conv_block(stage2_part3_branch3, 160, kernel=(7, 1))
    stage2_part3_branch3 = conv_block(stage2_part3_branch3, 192, kernel=(1, 7))

    stage2_part3_branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(output5)
    stage2_part3_branch4 = conv_block(stage2_part3_branch4, 192, 1, 1)

    output6 = layers.concatenate(
        [stage2_part3_branch1, stage2_part3_branch2, stage2_part3_branch3, stage2_part3_branch4], axis=3)

    # 第2阶段->第4部分(与第3部分一样)
    stage2_part4_branch1 = conv_block(output6, 192, kernel=(1, 1))

    stage2_part4_branch2 = conv_block(output6, 160, kernel=(1, 1))
    stage2_part4_branch2 = conv_block(stage2_part4_branch2, 160, kernel=(1, 7))
    stage2_part4_branch2 = conv_block(stage2_part4_branch2, 192, kernel=(7, 1))

    stage2_part4_branch3 = conv_block(output6, 160, kernel=(1, 1))
    stage2_part4_branch3 = conv_block(stage2_part4_branch3, 160, kernel=(7, 1))
    stage2_part4_branch3 = conv_block(stage2_part4_branch3, 160, kernel=(1, 7))
    stage2_part4_branch3 = conv_block(stage2_part4_branch3, 160, kernel=(7, 1))
    stage2_part4_branch3 = conv_block(stage2_part4_branch3, 192, kernel=(1, 7))

    stage2_part4_branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(output6)
    stage2_part4_branch4 = conv_block(stage2_part4_branch4, 192, 1, 1)

    output7 = layers.concatenate(
        [stage2_part4_branch1, stage2_part4_branch2, stage2_part4_branch3, stage2_part4_branch4], axis=3)

    # 第2阶段->第5部分
    stage2_part5_branch1 = conv_block(output7, 192, kernel=(1, 1))

    stage2_part5_branch2 = conv_block(output7, 192, kernel=(1, 1))
    stage2_part5_branch2 = conv_block(stage2_part5_branch2, 192, kernel=(1, 7))
    stage2_part5_branch2 = conv_block(stage2_part5_branch2, 192, kernel=(7, 1))

    stage2_part5_branch3 = conv_block(output7, 192, kernel=(1, 1))
    stage2_part5_branch3 = conv_block(stage2_part5_branch3, 192, kernel=(7, 1))
    stage2_part5_branch3 = conv_block(stage2_part5_branch3, 192, kernel=(1, 7))
    stage2_part5_branch3 = conv_block(stage2_part5_branch3, 192, kernel=(7, 1))
    stage2_part5_branch3 = conv_block(stage2_part5_branch3, 192, kernel=(1, 7))

    stage2_part5_branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(output7)
    stage2_part5_branch4 = conv_block(stage2_part5_branch4, 192, 1, 1)

    output8 = layers.concatenate(
        [stage2_part5_branch1, stage2_part5_branch2, stage2_part5_branch3, stage2_part5_branch4], axis=3)

    # 第3阶段(一共5部分)

    # 第3阶段->第1部分
    stage3_part1_branch1 = conv_block(output8, 192, kernel=(1, 1))
    stage3_part1_branch1 = conv_block(stage3_part1_branch1, 320, kernel=(3, 3), strides=(2, 2), padding='valid')

    stage3_part1_branch2 = conv_block(output8, 192, kernel=(1, 1))
    stage3_part1_branch2 = conv_block(stage3_part1_branch2, 192, kernel=(1, 7))
    stage3_part1_branch2 = conv_block(stage3_part1_branch2, 192, kernel=(7, 1))
    stage3_part1_branch2 = conv_block(stage3_part1_branch2, 192, kernel=(3, 3), strides=(2, 2), padding='valid')

    stage3_part1_branch3 = MaxPooling2D((3, 3), strides=(2, 2))(output8)

    output9 = layers.concatenate(
        [stage3_part1_branch1, stage3_part1_branch2, stage3_part1_branch3], axis=3)

    # 第3阶段->第2部分
    stage3_part2_branch1 = conv_block(output9, 320, kernel=(1, 1))

    stage3_part2_branch2 = conv_block(output9, 384, kernel=(1, 1))
    stage3_part2_branch2_1 = conv_block(stage3_part2_branch2, 384, kernel=(1, 3))
    stage3_part2_branch2_2 = conv_block(stage3_part2_branch2, 384, kernel=(3, 1))
    stage3_part2_branch2 = layers.concatenate(
        [stage3_part2_branch2_1, stage3_part2_branch2_2], axis=3)

    stage3_part2_branch3 = conv_block(output9, 448, kernel=(1, 1))
    stage3_part2_branch3 = conv_block(stage3_part2_branch3, 384, kernel=(3, 3))
    stage3_part2_branch3_1 = conv_block(stage3_part2_branch3, 384, kernel=(1, 3))
    stage3_part2_branch3_2 = conv_block(stage3_part2_branch3, 384, kernel=(3, 1))
    stage3_part2_branch3 = layers.concatenate(
        [stage3_part2_branch3_1, stage3_part2_branch3_2], axis=3)

    stage3_part2_branch4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(output9)
    stage3_part2_branch4 = conv_block(stage3_part2_branch4, 192, kernel=(1, 1))

    output10 = layers.concatenate(
        [stage3_part2_branch1, stage3_part2_branch2, stage3_part2_branch3, stage3_part2_branch4], axis=3)

    # 第3阶段->第3部分(与第2部分一样)
    stage3_part3_branch1 = conv_block(output10, 320, kernel=(1, 1))

    stage3_part3_branch2 = conv_block(output10, 384, kernel=(1, 1))
    stage3_part3_branch2_1 = conv_block(stage3_part3_branch2, 384, kernel=(1, 3))
    stage3_part3_branch2_2 = conv_block(stage3_part3_branch2, 384, kernel=(3, 1))
    stage3_part3_branch2 = layers.concatenate(
        [stage3_part3_branch2_1, stage3_part3_branch2_2], axis=3)

    stage3_part3_branch3 = conv_block(output10, 448, kernel=(1, 1))
    stage3_part3_branch3 = conv_block(stage3_part3_branch3, 384, kernel=(3, 3))
    stage3_part3_branch3_1 = conv_block(stage3_part3_branch3, 384, kernel=(1, 3))
    stage3_part3_branch3_2 = conv_block(stage3_part3_branch3, 384, kernel=(3, 1))
    stage3_part3_branch3 = layers.concatenate(
        [stage3_part3_branch3_1, stage3_part3_branch3_2], axis=3)

    stage3_part3_branch4 = AveragePooling2D((3, 3),  strides=(1, 1), padding='same')(output10)
    stage3_part3_branch4 = conv_block(stage3_part3_branch4, 192, kernel=(1, 1))

    output10 = layers.concatenate(
        [stage3_part3_branch1, stage3_part3_branch2, stage3_part3_branch3, stage3_part3_branch4], axis=3)

    # avg_pool + FC
    output_final = GlobalAveragePooling2D(name='avg_pool')(output10)
    output_final = Dense(classes, activation='softmax', name='predictions')(output_final)

    inputs = img_input

    model = Model(inputs, output_final)

    return model


# 图片像素归一化到(-1, 1)
def preprocess_input(input_feature):
    output = (input_feature / 255 - 0.5) * 2

    return output


if __name__ == '__main__':

    model = inceptionV3()
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    # 读取图片并整形(224, 224, 3)
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 预测并找到预测概率最大的类别作为结果
    predict = model.predict(x)
    print(np.argmax(predict))
    print('Predicted:', decode_predictions(predict))

