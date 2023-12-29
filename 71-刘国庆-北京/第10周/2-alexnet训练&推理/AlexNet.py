# 导入 Keras 库中的一些层（layers）
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
# 导入 Keras 库中的 Sequential 类,用于按顺序构建神经网络模型
from keras.models import Sequential


# 定义 AlexNet 函数,指定输入形状(224, 224, 3)和输出形状的默认值2
def AlexNet(input_shape=(224, 224, 3), out_shape=2):
    # 创建 Sequential 模型
    model = Sequential()

    # 添加第一层卷积层：
    # 输出特征层的深度filters为 48
    # 使用kernel_size大小为 (11, 11),strides步长为 (4, 4)的卷积核对图像进行卷积,
    # 使用 'valid' 填充方式,
    # 输入形状input_shape为 input_shape,激活函数activation为 'relu'。
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )

    # 添加批量归一化层
    model.add(BatchNormalization())

    # 添加第一层最大池化层：
    # 使用大小pool_size为 (3, 3) 的池化窗口,步长strides为 (2, 2) 进行最大池化,
    # 输出特征图的形状为 (27, 27, 96)。
    # 使用 'valid' 填充方式padding。
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 20),
            padding='valid'
        )
    )

    # 添加第二层卷积层：
    # 输出特征层的深度filters为 128
    # 使用大小kernel_size为 (5, 5),步长strides为 (1, 1)的卷积核对图像进行卷积,
    # 使用 'same' 填充方式padding,激活函数activation为 'relu'。
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 添加批量归一化层
    model.add(BatchNormalization())

    # 添加第二层最大池化层：
    # 使用大小pool_size为 (3, 3) 的池化窗口,步长strides为 (2, 2) 进行最大池化,
    # 使用 'valid' 填充方式padding。
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 添加第三层卷积层：
    # 输出特征层的深度filters为 192
    # 使用大小kernel_size为 (3, 3),步长strides为 (1, 1)的卷积核对图像进行卷积,
    # 使用 'same' 填充方式padding,激活函数activation为 'relu'
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 添加第四层卷积层：
    # 输出特征层的深度filters为 192
    # 使用大小kernel_size为 (3, 3),步长strides为 (1, 1)的卷积核对图像进行卷积,
    # 使用 'same' 填充方式padding,激活函数activation为 'relu'
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 添加第五层卷积层：
    # 输出特征层的深度filters为 128
    # 使用大小kernel_size为 (3, 3),步长strides为 (1, 1)的卷积核对图像进行卷积,
    # 使用 'same' 填充方式padding,激活函数activation为 'relu'。
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 添加第三层最大池化层：
    # 使用大小pool_size为 (3, 3) 的池化窗口,步长strides为 (2, 2) 进行最大池化,
    # 使用 'valid' 填充方式padding。
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 添加两个全连接层：

    # 将之前卷积层输出的特征图展平为一维向量。
    model.add(Flatten())

    # 添加第一个全连接层：
    # 输出维度为 1024,激活函数为 'relu'。
    model.add(Dense(1024, activation='relu'))
    # 添加 Dropout 层,防止过拟合,丢弃率为 25%。
    model.add(Dropout(0.25))

    # 添加第二个全连接层：
    # 输出维度为 1024,激活函数为 'relu'。
    model.add(Dense(1024, activation='relu'))
    # 添加 Dropout 层,防止过拟合,丢弃率为 25%。
    model.add(Dropout(0.25))

    # 添加输出层：
    # 输出维度为 output_shape,激活函数为 'softmax'。
    # 在这个例子中,输出为2类。
    model.add(Dense(out_shape, activation='softmax'))
    # 返回构建好的神经网络模型
    return model
