import tensorflow as tf
# 加载mnist训练集和测试集
(train_images, train_label), (test_images, test_label) = tf.keras.datasets.mnist.load_data()


# 处理数据
train_images = train_images.reshape(60000, (28*28))
test_images = test_images.reshape(10000, (28*28))
# 归一化
train_images = train_images / 255.0
test_images = test_images / 255.0
# one-hot
train_label = tf.keras.utils.to_categorical(train_label)
test_label = tf.keras.utils.to_categorical(test_label)

# 构建神经网络结构
network = tf.keras.models.Sequential()
network.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(tf.keras.layers.Dense(10, activation='softmax'))
# 配置编译模型
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
print(train_images.shape, train_label.shape)
network.fit(train_images, train_label, epochs=5, batch_size=128)

# 测试模型效果
loss, metrics = network.evaluate(test_images, test_label, verbose=2)
# 打印loss和准确度
print(loss, metrics)

# 推理
res = network.predict(test_images)
print(res)
