"""
keras实现手写数字的识别
"""

[1]
"""将训练数据和测试数据加载到内存中"""
from tensorflow.keras.datasets import mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
print("train_images_shape:", train_images.shape)
print("train_labels:",train_labels)
print("test_images_shape:",test_images.shape)
print("test_labels:",test_labels)

[2]
"""预览测试数据第一个数字（为7）"""
digit = test_images[0]
import matplotlib.pyplot as plt
#使用matplotlib库的cm子库中的binary,图像被渲染成黑白
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()

[3]
"""使用tensorflow.Keras搭建有效识别图案的神经网络"""
from tensorflow.keras import models
from tensorflow.keras import layers

#创建一个简单的Sequential模型(顺序模型，但可以构造非常复杂的网络模型)
#可以把每个数据处理层串联起来
network = models.Sequential()

"""
layers.Dense()：构造一个数据处理层
input_shape:输入像素点；activation:激活函数；512：输出（隐藏层512个结点）
input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的","表示数组里面的每一个元素到底包含多少个数字都没有关系."""

#全连接层
network.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
#输出层
network.add(layers.Dense(10 , activation="softmax"))

"""
使用RMSprop优化器、分类交叉熵损失函数和准确度来编译神经网络模型
RMSprop优化器：基于梯度的优化算法，用于更新神经网络的权重以最小化损失函数
loss=...：指定分类交叉熵损失函数，适用多类别分类问题。
metrics=...：指定评估指标为准确率"""
network.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                metrics=['accuracy'])

[4]
"""
将图像的二维形式转化为一维形式，进行归一化
将训练图像和测试图像的标签独热编码
"""

#将二维数组转化为一维数组，由于灰度图像素点大小为整数，进行归一化
train_images = train_images.reshape((60000,28*28)).astype("float32")/255
test_images = test_images.reshape((10000,28*28)).astype('float32')/255

"""
to_categorical自动转为10个大小的数组
将图片对应的标记进行更改，图片为0-9
如测试第一个为7，则改变后为[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]（one hot独热编码）
"""
from tensorflow.keras.utils import to_categorical
print("before change:",test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change:",test_labels[0])

[5]
"""
将数据输入网络训练
输入：手写数字图片；输出：图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算
epochs：每次计算循环次数为5
"""

network.fit(train_images, train_labels, epochs=5, batch_size=128)

[6]
"""
测试数据输入，检验网络训练后的图片识别效果
识别效果与硬件有关
"""
#verbose:指定了输出评估过程的详细程度，其中1表示输出进度条和指标信息。
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print("test_loss:",test_loss)
print("test_acc:",test_acc)

[7]
"""
输入一张手写图片到网络中，看看它的识别情况
"""
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
digit = test_images[2]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()

test_images = test_images.reshape((10000,28*28))
res = network.predict(test_images)
print("res[2]:",res[2])
print("res[2].shape:",res[2].shape)
for i in range(res[2].shape[0]):
    if(res[2][i] == 1):
        print("the number for the predict is :",i)
        break