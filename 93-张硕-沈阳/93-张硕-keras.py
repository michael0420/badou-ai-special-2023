#加载数据

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape=', train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape=', test_images.shape)
print('test_labels = ', test_labels)

#打印第一张
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

#搭建神经网络
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,))) #隐藏层
network.add(layers.Dense(10, activation='softmax'))  #输出层

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#归一化
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

#one-hot
from tensorflow.keras.utils import to_categorical
print('before change:', test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('after change:', test_labels[0])

#训练
network.fit(train_images, train_labels, epochs = 5, batch_size = 128)

#测试数据
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

#推理
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("推理出来的图片数字是：", i)
        break

import cv2
test6 = cv2.imread('test6.png', 0)
test6 = test6.reshape((1,28*28))
test6 = test6.astype('float32')/255
print(test6.shape)
res1 = network.predict(test6)
print('res1\n', res1)

