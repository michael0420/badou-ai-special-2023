#!/usr/bin/env python
# encoding=utf-8
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical


(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
# network.add(layers.Dense(256, activation="relu", input_shape=(512,)))
network.add(layers.Dense(10, activation="softmax"))
# metrics=["accuracy"] 不写会导致evaluate运行出错
network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# 处理图片
train_data = train_data.reshape((60000, 28*28))
train_data = train_data.astype("float32") / 255
test_data = test_data.reshape((10000, 28*28))
test_data = test_data.astype("float32") / 255

# 处理标签
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练
network.fit(train_data, train_labels, epochs=5, batch_size=128)
# 测试
print(test_data.shape, test_labels.shape)
test_loss, test_acc = network.evaluate(test_data, test_labels)
print("测试集loss:", test_loss)
print("测试集accuracy: ", test_acc)

# 推理
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
test_data = test_data.reshape((10000, 28*28))
res = network.predict(test_data)
print("第一个数字是: ", np.argmax(res[0]))
