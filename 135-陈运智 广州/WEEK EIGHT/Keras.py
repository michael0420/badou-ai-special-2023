# 获取数据
# train_images是用于训练系统得手写数字图片
# train_labels是用于标注图片信息
# test_images是用于检测系统训练效果的图片
# test_labels是test_images对应的数字标签
import numpy as np
from tensorflow.keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels) =mnist.load_data()
print('train_images.shape =' ,train_images.shape)
print('train_labels=',train_labels)
print('test_images.shape=',test_images.shape)
print('test_labels=',test_labels)

# 打印测试集里面的第一张图片
testFirstImage=test_images[0]
import matplotlib.pyplot as plt
# 将数据映射到黑白颜色范围
plt.imshow(testFirstImage,cmap=plt.cm.binary)
# 显示颜色条
plt.colorbar()
plt.show()

# 构建keras神经网络
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(5,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# 测试集进行调整，将每个图像的二维数组（28x28像素）拉平为一个一维数组（28*28=784个元素）
# 归一化处理将像素值缩放到 0 到 1 之间
train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255


from tensorflow.keras.utils import to_categorical
print("before change",train_labels[0])
train_labels =to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change",train_labels[0])

network.fit(train_images,train_labels,epochs=2,batch_size=128)
test_loss,test_acc =network.evaluate(test_images,test_labels)
(train_images,train_labels),(test_images,test_labels) =mnist.load_data()
testFirstImage=test_images[0]
import matplotlib.pyplot as plt
# 将数据映射到黑白颜色范围
plt.imshow(testFirstImage,cmap=plt.cm.binary)
# 显示颜色条
plt.colorbar()
plt.show()
test_images = test_images.reshape((10000, 28*28))
# test_images = test_images.astype('float32') / 255
# test_img=test_images[0].reshape(784,)
# predictions=network.predict(np.array([test_img]))
# # 找到概率最高的类别的索引
# predicted_class_index = np.argmax(predictions)
# # 输出预测结果
# print("Predicted Class Index:", predicted_class_index)
res=network.predict(test_images)
print(res)