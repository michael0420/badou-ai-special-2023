[1]
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)                                       #读取数据

[2]
digit = test_images[0]                                                  #测试图片打印
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

[3]
from tensorflow.keras import models                                     #建立简单的神经网络
from tensorflow.keras import layers

network = models.Sequential()                                           #把每个数据层串联起来
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))     #构造一个全连接层，512个节点隐藏层，relu是激活函数，
network.add(layers.Dense(10, activation='softmax'))                         #构造一个全连接层，10个节点输出层，

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])
      
[4]
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255                         #转变像素点为【0,1】直接的浮点值

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255                           #转变像素点为【0,1】直接的浮点值


from tensorflow.keras.utils import to_categorical
print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)                                 #将数据输入网络进行训练
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

[5]
network.fit(train_images, train_labels, epochs=5, batch_size = 128)         #每5次进行循环，以128个作为一组计算

[6]
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)     #检验识别效果
print(test_loss) 
print('test_acc', test_acc)

[7]
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()    #导入图片查看识别效果
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break

