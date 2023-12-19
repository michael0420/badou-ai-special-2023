# 导入 TensorFlow 中的 Keras 模块并从中导入 MNIST 数据集
from tensorflow.keras.datasets import mnist
# 导入 matplotlib.pyplot 模块用于绘图
import matplotlib.pyplot as plt
# 从 TensorFlow 的 Keras 模块中导入模型
from tensorflow.keras import models
# 从 TensorFlow 的 Keras 模块中导入层
from tensorflow.keras import layers
# 从 TensorFlow 的 Keras 工具中导入 to_categorical 函数
from tensorflow.keras.utils import to_categorical

# 1,将训练数据和检测数据加载到内存中
# 使用 mnist.load_data() 从 MNIST 数据集中加载训练和测试数据
# 将训练数据集的图像存储在 train_images 中,训练标签存储在 train_labels 中
# 将测试数据集的图像存储在 test_images 中,测试标签存储在 test_labels 中
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 打印训练图像的形状,即图像的维度信息
print(f"训练图像的形状:{train_images.shape}")
# 打印训练标签,即与训练图像对应的手写数字的真实值
print(f"训练标签:{train_labels}")
# 打印测试图像的形状,用于了解测试数据的结构和维度
print(f"测试图像的形状:{test_images.shape}")
# 打印测试标签,即与测试图像对应的手写数字的真实值
print(f"测试标签:{test_labels}")

# 2,打印用于测试的第一张图片
# 从测试图像中选择第一张图像
digit = test_images[0]
# 使用 plt.imshow() 绘制灰度图像,cmap=plt.cm.binary 表示使用二进制颜色映射
plt.imshow(digit, cmap=plt.cm.binary)
# 显示图像
plt.show()

# 3,使用tensorflow.Keras搭建一个有效识别图案的神经网络
# 创建一个顺序模型network
network = models.Sequential()
# 向模型中添加一个全连接层（Dense）：512个神经元,激活函数为 ReLU,输入形状为 (28 * 28,)
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# 向模型中添加一个全连接层：10个神经元,激活函数为 Softmax
network.add(layers.Dense(10, activation='softmax'))
# 编译模型,设置优化器为 'rmsprop',损失函数为 'categorical_crossentropy',度量标准为 'accuracy'
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 4,在把数据输入到网络模型之前,把数据做归一化处理
# 重新塑形训练图像数组,将每张图像转换为一维数组（28 * 28 = 784）
train_images = train_images.reshape((60000, 28 * 28))
# 将训练图像数组中的数据类型转换为 float32
train_images = train_images.astype('float32')
# 将像素值缩放到 0 到 1 之间,通过除以 255
train_images = train_images / 255
# 重新塑形测试图像数组,将每张图像转换为一维数组（28 * 28 = 784）
test_images = test_images.reshape((10000, 28 * 28))
# 将测试图像数组中的数据类型转换为 float32
test_images = test_images.astype('float32')
# 将像素值缩放到 0 到 1 之间,通过除以 255
test_images = test_images / 255

# 5,one-hot
# 打印改变之前的测试标签的值
print(f"改变之前测试标签值:{test_labels}")
# 使用 to_categorical 方法将训练标签转换为 one-hot 编码
train_labels = to_categorical(train_labels)
# 使用 to_categorical 方法将测试标签转换为 one-hot 编码
test_labels = to_categorical(test_labels)
# 打印改变之后的测试标签的值
print(f"改变之后测试标签值:{test_labels}")

# 6,数据输入网络,使用神经网络模型进行训练
# train_images 是训练图像数据,train_labels 是对应的训练标签
# epochs=5 表示迭代训练 5 轮,batch_size=128 表示每批次使用 128 个样本
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 7,测试数据输入,检验网络学习后的图片识别效果(使用神经网络模型评估测试数据集)
# test_images 是测试图像数据,test_labels 是对应的测试标签,verbose=1 表示输出详细信息
test_loss, test_accuracy = network.evaluate(test_images, test_labels, verbose=1)
# 打印测试损失值
print(f"测试损失值:{test_loss}")
# 打印测试准确率
print(f"测试准确率:{test_accuracy}")

# 8,输入一张手写数字图片到网络中,看看它的识别效果
# 从 MNIST 数据集中加载训练和测试数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 选择测试数据集中的第二张图像
digit_2 = test_images[1]
# 使用 plt.imshow() 绘制灰度图像,cmap=plt.cm.binary 表示使用二进制颜色映射
plt.imshow(digit_2, cmap=plt.cm.binary)
# 显示图像
plt.show()
# 重新塑形测试图像数组,将每张图像转换为一维数组（28 * 28 = 784）
test_images = test_images.reshape((10000, 28 * 28))
# 使用神经网络模型进行预测
result = network.predict(test_images)
# 遍历预测结果的第二个类别（索引为 1）
for i in range(result[1].shape[0]):
    if result[1][i] == 1:
        # 打印预测结果为数字 i
        print(f"图片的编号是{i}")
        break
