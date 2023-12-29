# 该文件负责读取Cifar-10数据并对其进行数据增强预处理
# 导入操作系统模块
import os
# 导入 TensorFlow 深度学习框架，并将其命名为 tf
import tensorflow as tf

# 设置类别的数量为 10
num_classes = 10
# 设定用于训练的每个周期的样本总数为 50000
num_examples_pre_epoch_for_train = 50000
# 设定用于评估的每个周期的样本总数为 10000
num_examples_pre_epoch_for_eval = 10000


# 定义一个空类，用于返回读取的Cifar-10的数据
class CIFAR10Record(object):
    pass


# 定义一个读取Cifar-10的函数read_cifar10()，这个函数的目的就是读取目标文件里面的内容
def read_cifar10(file_queue):
    # 创建一个CIFAR10Record实例，用于存储读取到的数据
    result=CIFAR10Record
    # 设定标签所占字节数，如果是Cifar-100数据集，则此处为2
    label_bytes=1
    # 图片高度
    result.height=32
    # 图片宽度
    result.width=32
    # 图片深度（RGB三通道）
    result.channels=3
    # 计算图片样本总元素数量
    image_bytes=result.height*result.width*result.channels
    # 计算每个样本的总字节数（包含标签和图片）
    record_bytes=image_bytes+label_bytes
    # 使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的是读取文件
    reader=tf.FixedLenRecordReader(record_bytes=record_bytes)
    # 使用该类的read()函数从文件队列里面读取文件
    result.key,value=reader.read(file_queue)
    # 读取到文件后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
    record_bytes=tf.decode_raw(value,tf.uint8)
    # 将数组的第一个元素作为标签，使用tf.strided_slice()函数将标签提取出来
    # 使用tf.cast()函数将这一个标签转换成int32的数值形式
    result.label=tf.cast(tf.stied_slice(record_bytes,[0],[label_bytes]),tf.int32)
    # 剩下的元素是图片数据，将一维数据转换成3维数据，形状为[depth, height, width]
    channel_major=tf.cast(tf.stried_slice(record_bytes,[label_bytes],[label_bytes+image_bytes]),[result.channels,result.height,result.width])
    # 转换数据排布方式，变为(height, width, depth)
    result.uint8image=tf.transpose(channel_major,[1,2,0])
    # 返回已经读取出的数据存储实例
    return result


# 这个函数对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
def inputs(data_dir, batch_size, distorted):
    # 拼接文件地址
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    # 根据已有的文件地址创建一个文件队列
    file_queue = tf.train.string_input_producer(filenames)
    # 使用已定义的文件读取函数read_cifar10()读取队列中的文件
    read_input = read_cifar10(file_queue)
    # 将已经转换好的图片数据再次转换为float32的形式
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    # 获取训练集样本总数
    num_examples_per_epoch = num_examples_pre_epoch_for_train
    if distorted != None:  # 如果distorted参数不为空值，代表要进行图片增强处理
        # 将预处理好的图片进行剪切
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
        # 将剪切好的图片进行左右翻转
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        # 将左右翻转好的图片进行随机亮度调整
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
        # 将亮度调整好的图片进行随机对比度调整
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        # 进行标准化图片操作，对每一个像素减去平均值并除以像素方差
        float_image = tf.image.per_image_standardization(adjusted_contrast)
        # 设置图片数据及标签的形状
        # 设置图片数据的形状为 [24, 24, 3]
        float_image.set_shape([24, 24, 3])
        # 设置标签数据的形状为 [1]
        read_input.label.set_shape([1])
        # 计算用于训练的最小队列样本数
        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        # 打印提示信息，填充队列以确保在开始训练之前具有足够的 CIFAR 图像。这可能需要几分钟时间。
        print(
            "Filling queue with %d CIFAR images before starting to train. This will take a few minutes." % min_queue_examples)
        # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label
        # 使用tf.train.shuffle_batch()函数随机产生一个包含训练数据的 batch
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples,
                                                            )
        # 返回训练数据的图片和标签，标签进行形状调整
        return images_train, tf.reshape(labels_train, [batch_size])
    else:  # 不对图像数据进行数据增强处理
        # 在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        # 剪切完成以后，直接进行图片标准化操作
        # 进行每张图片的标准化操作
        float_image = tf.image.per_image_standardization(resized_image)
        # 设置标准化后的图片数据的形状为 [24, 24, 3]
        float_image.set_shape([24, 24, 3])
        # 设置标签数据的形状为 [1]
        read_input.label.set_shape([1])
        # 计算用于训练的最小队列样本数
        min_queue_examples = int(num_examples_per_epoch * 0.4)
        # 使用tf.train.batch()函数生成包含测试数据的 batch
        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)
        # 返回测试数据的图片和标签，标签进行形状调整
        return images_test, tf.reshape(labels_test, [batch_size])
