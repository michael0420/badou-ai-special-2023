# 导入将在未来版本中成为内置功能的特性，确保在 Python 2 和 Python 3 中都能使用 print 函数的语法
from __future__ import print_function
# 导入 NumPy 库，用于处理数组和矩阵等数学运算
import numpy as np
# 导入 Keras 模块中的各种神经网络层
from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
# 导入 Keras 模块中的其他层，如激活函数、批标准化层和展平层
from keras.layers import Activation, BatchNormalization, Flatten
# 导入 Keras 中的 Model 类，用于构建神经网络模型
from keras.models import Model
# 导入 Keras 中的图像预处理模块，用于对图像进行预处理
from keras.preprocessing import image
# 导入 Keras 中的后端模块，允许直接访问底层张量操作
import keras.backend as K
# 导入 Keras 工具函数，用于获取文件
from keras.utils.data_utils import get_file
# 导入 Keras 应用程序模块中的函数，用于在 ImageNet 数据集上进行预训练模型的预测结果解码和输入预处理
from keras.applications.imagenet_utils import decode_predictions, preprocess_input


# 定义残差块函数，用于构建恒等映射残差块
def identity_block(input_tensor, kernel_size, filters, stage, block):
    # 从 filters 中获取每个卷积层的滤波器数量
    filters1, filters2, filters3 = filters

    # 定义卷积层和批标准化的基本名称
    # 创建卷积层的基本名称，基于给定的阶段和块的信息
    conv_name_base = 'res' + str(stage) + block + '_branch'
    # 创建批标准化层的基本名称，基于给定的阶段和块的信息
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 第一个卷积层，1x1 卷积，使用 filters1 个滤波器，命名为 conv_name_base + '2a'
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    # 批标准化层，命名为 bn_name_base + '2a'
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 第二个卷积层，使用指定大小的卷积核，padding 为 'same' 表示使用零填充
    # 滤波器数量为 filters2，命名为 conv_name_base + '2b'
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    # 批标准化层，命名为 bn_name_base + '2b'
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 第三个卷积层，1x1 卷积，使用 filters3 个滤波器，命名为 conv_name_base + '2c'
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    # 批标准化层，命名为 bn_name_base + '2c'
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    # 将输入张量与卷积结果相加，实现残差连接
    x = layers.add([x, input_tensor])
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 返回构建好的残差块
    return x


# 定义卷积块函数，用于构建带有卷积和短路连接的残差块
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    # 从 filters 中获取每个卷积层的滤波器数量
    filters1, filters2, filters3 = filters

    # 定义卷积层和批标准化的基本名称
    # 创建卷积层的基本名称，基于给定的阶段和块的信息
    conv_name_base = 'res' + str(stage) + block + '_branch'
    # 创建批标准化层的基本名称，基于给定的阶段和块的信息
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 第一个卷积层，1x1 卷积，使用 strides 进行步幅设置
    # 滤波器数量为 filters1，命名为 conv_name_base + '2a'
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    # 批标准化层，命名为 bn_name_base + '2a'
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 第二个卷积层，使用指定大小的卷积核，padding 为 'same' 表示使用零填充
    # 滤波器数量为 filters2，命名为 conv_name_base + '2b'
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    # 批标准化层，命名为 bn_name_base + '2b'
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 第三个卷积层，1x1 卷积，使用 filters3 个滤波器，命名为 conv_name_base + '2c'
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    # 批标准化层，命名为 bn_name_base + '2c'
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    # 短路连接，使用 1x1 卷积进行卷积操作，步幅为 strides
    # 短路连接，使用 1x1 卷积进行卷积操作，步幅为 strides
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    # 对短路连接结果应用批标准化，其名称为 bn_name_base + '1'
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
    # 将卷积结果与短路连接相加，实现残差连接
    x = layers.add([x, shortcut])
    # 使用激活函数 'relu' 进行非线性变换
    x = Activation('relu')(x)

    # 返回构建好的卷积块
    return x


def ResNet50(input_shape=[224, 224, 3], classes=1000):
    # 创建输入层，指定输入图像的形状
    img_input = Input(shape=input_shape)

    # 对输入图像进行零填充，防止特征图尺寸缩小太快
    x = ZeroPadding2D((3, 3))(img_input)
    # 第一层卷积操作，使用64个7x7的卷积核，步幅为2，生成特征图
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    # 对卷积结果进行批量归一化
    x = BatchNormalization(name='bn_conv1')(x)
    # 使用ReLU激活函数激活卷积结果
    x = Activation('relu')(x)
    # 最大池化操作，窗口大小为3x3，步幅为2
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # ResNet块，阶段2
    # 第一块（标记为 'a'）
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    # 第二块（标记为 'b'）
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    # 第三块（标记为 'c'）
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # ResNet块，阶段3
    # 第一块（标记为 'a'）
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    # 第二块（标记为 'b'）
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    # 第三块（标记为 'c'）
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    # 第四块（标记为 'd'）
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # ResNet块，阶段4
    # 第一块（标记为 'a'）
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    # 第二块（标记为 'b'）
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    # 第三块（标记为 'c'）
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    # 第四块（标记为 'd'）
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    # 第五块（标记为 'e'）
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    # 第六块（标记为 'f'）
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # ResNet块，阶段5
    # 第一块（标记为 'a'）
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    # 第二块（标记为 'b'）
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    # 第三块（标记为 'c'）
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 全局平均池化
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    # 展平操作
    x = Flatten()(x)
    # 全连接层，输出为指定的类别数量，使用softmax激活函数
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    # 创建ResNet50模型
    model = Model(img_input, x, name='resnet50')
    # 加载预训练的权重
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    # 返回创建的模型
    return model


# 如果当前脚本是主程序（而不是被导入的模块），则执行以下代码
if __name__ == '__main__':
    # 创建ResNet50模型
    model = ResNet50()
    # 打印模型摘要
    model.summary()
    # 读取图像文件路径
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    # 使用Keras的image模块加载图像，调整大小为(224, 224)
    img = image.load_img(img_path, target_size=(224, 224))
    # 将图像转换为NumPy数组
    x = image.img_to_array(img)
    # 在数组的第一维度上添加一个维度，使其成为形状为(1, 224, 224, 3)的张量
    x = np.expand_dims(x, axis=0)
    # 对图像进行预处理，以适应ResNet50模型的输入要求
    x = preprocess_input(x)
    # 打印输入图像的形状
    print('Input image shape:', x.shape)
    # 使用模型进行预测
    preds = model.predict(x)
    # 打印预测结果（使用ImageNet类标签解码）
    print('Predicted:', decode_predictions(preds))
