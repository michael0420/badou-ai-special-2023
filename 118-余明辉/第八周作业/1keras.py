from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
#导入数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#搭建网络
from tensorflow.keras import models
from tensorflow.keras import layers

net = models.Sequential()
net.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
net.add(layers.Dense(10,activation='relu'))

net.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])

#处理数据，（降维，归一化）

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

#处理结果数据
from tensorflow.keras.utils import to_categorical
print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

#开始训练
net.fit(train_images, train_labels, epochs=5, batch_size = 128)

#查看误差，检验模型
test_loss, test_acc = net.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = net.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break