# 导入 Keras 库中的一些回调函数,用于在训练神经网络时执行特定的操作
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# 导入 Keras 中的一个实用工具,用于处理 NumPy 数组的工具函数
from keras.utils import np_utils
# 导入 Keras 中的 Adam 优化器,用于更新神经网络的权重
from keras.optimizers import Adam
# 导入一个自定义的模型类 AlexNet
import AlexNet
# 导入 NumPy 库,并将其重命名为 np,用于处理数组和矩阵
import numpy as np
# 导入名为 utils 的模块或包,其中可能包含一些与实用函数相关的工具,具体内容需要在代码中查看
import utils
# 导入 OpenCV 库,用于计算机视觉任务,比如图像处理和计算机视觉算法
import cv2
# 导入 Keras 的 backend 模块,并将其重命名为 K,提供对底层深度学习框架的访问


# 使用 Keras 的 backend 模块设置图像维度的顺序为 TensorFlow 风格


# 定义数据生成器函数,用于批量生成训练数据
def generate_arrays_from_file(lines, batch_size):
    # 获取数据行的总长度
    n = len(lines)
    # 初始化索引 i 为 0,用于迭代访问数据行
    i = 0
    # 使用无限循环,确保能够持续生成数据
    while 1:
        # 初始化用于存储图像数据的列表 X_train
        X_train = []
        # 初始化用于存储标签数据的列表 Y_train
        Y_train = []
        # 获取一个 batch_size 大小的数据
        for b in range(batch_size):
            # 如果达到数据末尾,重新打乱数据顺序
            if i == 0:
                np.random.shuffle(lines)
            # 从文件行数据中提取文件名和标签
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            # 从指定路径读取图像文件并保存到变量 img
            img = cv2.imread(r".\data\image\train" + '/' + name)
            # 将图像颜色通道顺序从 BGR 转换为 RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 将图像归一化处理
            # 将图像像素值进行归一化处理,将像素值范围缩放到 [0, 1]
            img = img / 255
            # 将处理后的图像添加到 X_train 列表中
            X_train.append(img)
            # 读取并保存图像对应的标签
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i + 1) % n
        # 处理图像
        # 使用自定义的 resize_image 函数对图像列表 X_train 进行大小调整
        X_train = utils.resize_image(X_train, (224, 224))
        # 将调整大小后的图像列表转换为指定形状(-1, 224, 224, 3)的 NumPy 数组
        X_train = X_train.reshape(-1, 224, 224, 3)
        # 将标签转换为独热编码格式
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        # 使用 yield 关键字返回批量的训练数据
        yield X_train, Y_train


# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 设置模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    # 使用只读模式打开指定文件,读取文件内容到变量 lines
    with open(r".\data\dataset.txt", "r") as f:
        # 读取文件中的所有行并保存到变量 lines
        lines = f.readlines()

    # 打乱行,这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    # 设置随机种子以确保可重复性
    np.random.seed(10101)
    # 打乱数据行的顺序
    np.random.shuffle(lines)
    # 重置随机种子,使其不再受限于之前设置的值
    np.random.seed(None)

    # 90%用于训练,10%用于估计。
    # 计算验证集的样本数量,占总数据行的 10%
    num_val = int(len(lines) * 0.1)
    # 计算训练集的样本数量,剩余的 90%
    num_train = len(lines) - num_val
    # 建立AlexNet模型
    model = AlexNet.AlexNet()

    # 创建 ModelCheckpoint 回调函数,用于定期保存模型权重,放在checkpoint_period1
    # 定义保存模型的文件名格式,包括 epoch 数、训练损失和验证损失
    # 监控指标monitor为准确度acc
    # 保存整个模型而不仅仅是权重save_weights_only=False
    # 仅保存在验证集上性能最好的模型save_best_only=True
    # 每 3 个 epoch 保存一次
    checkpoint_period1 = ModelCheckpoint(
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        epoch=3
    )

    # 学习率下降的方式,acc三次不下降就下降学习率继续训练
    # 创建 ReduceLROnPlateau 回调函数,用于在训练过程中动态调整学习率reduce_lr
    # 监控指标monitor为准确度acc
    # 学习率减少的因子factor=0.5,新学习率 = 原学习率 * factor
    # 如果连续 3 个 epoch 准确度没有提升,则降低学习率patience=3
    # 输出学习率调整的信息verbose=1
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )

    # 是否需要早停,当val_loss一直不下降的时候意味着模型基本训练完毕,可以停止
    # 创建 EarlyStopping 回调函数,用于在训练过程中根据验证集损失来提前停止训练
    # 监控指标monitor为验证集损失val_loss
    # 不考虑损失的最小变化量min_delta=0
    # 如果连续 10 个 epoch 验证集损失没有提升,则停止训练patience=10
    # 输出提前停止训练的信息verbose=1
    early_stopping = EarlyStopping(
        monitor='acc',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 交叉熵
    # 编译模型,配置训练过程
    # 设置损失函数loss为分类交叉熵categorical_crossentropy
    # 设置优化器optimizer为Adam,学习率为0.001:lr=1e-3
    # 设置评估指标metrics为准确率accuracy
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-3),
        metrics=['accuracy']
    )

    # 一次的训练集大小
    batch_size = 128
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 使用生成器训练模型model.fit_generator
    # 训练数据的生成器函数generate_arrays_from_file,接受文件行列表lines[:num_train]和批量大小作为参数batch_size
    # 每个训练 epoch 中的步数steps_per_epoch=训练样本数num_train除以批量大小batch_size,确保至少为 1,
    # 用于验证的数据生成器函数,使用剩余的数据行作为验证数据validation_data
    # 验证步数validation_steps=验证样本数num_val 除以批量大小batch_size,确保至少为 1
    # 指定训练的总 epoch 数50
    # 指定从第一个 epoch 开始训练
    # 回调函数的列表,用于在训练过程中执行一些操作checkpoint_period1,reduce_lr
    model.fit_generator(
        generate_arrays_from_file(lines[:num_train], batch_size),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
        validation_steps=max(1, num_val // batch_size),
        epochs=50,
        initial_epoch=0,
        callbacks=[checkpoint_period1, reduce_lr]
    )

    # 在训练结束后,将模型的权重保存到一个HDF5文件中
    # 文件的路径为 log_dir + 'last1.h5'
    # HDF5 是一种用于存储大量数据的文件格式
    model.save_weights(log_dir + 'last1.h5')
