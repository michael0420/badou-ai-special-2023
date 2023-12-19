from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


# 1.加载训练模型(keras自带的mnist数据集)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2.设置网络架构，先设置一个空的顺序网络框架，再往里面加每层神经网络
# 网络架构：
# 3x3卷积(28, 28, 32) - 3x3卷积(28, 28, 8) -
# 全连接(28x28x4 - 512) - 全连接(28x28x4 - 64) -
# 全连接(28x28x4 - 10)
model = models.Sequential()
model.add(layers.Conv2D(32, 3, strides=(1, 1), padding="same", activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(8, 3, strides=(1, 1), padding="same", activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Flatten(input_shape=[28, 28, 8]))
model.add(layers.Dense(512, activation='relu', input_shape=(28*28*8,)))
model.add(layers.Dense(64, activation='relu', input_shape=(512,)))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
      
# 3.数据预处理。这里我用了卷积，所以直接输入即可

train_images = train_images.astype('float32') / 255
train_images = train_images.reshape((60000, 28, 28, 1))

test_images = test_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 4.训练
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# 5.测试
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('损失：', test_loss)
print('测试准确度：', test_acc)

# 6.展示 取第100张图片
digit = test_images[100]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28, 28, 1))
res = model.predict(test_images)
# one-hot结果中第几个为1，就是数字几
a = res[100]
for i in range(res[100].shape[0]):
    if res[100][i] == max(res[100]):
        print("the number for the picture is : ", i)
        break

