import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# 载入MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 将像素值标准化到 0 到 1 之间
x_train, x_test = x_train / 255.0, x_test / 255.0

# 对标签进行独热编码
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 显示第一张图片
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {np.argmax(y_train[0])}")  # 显示标签
plt.show()
# 构建模型
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # 将 28x28 的图像展平成一维数组
model.add(Dense(128, activation='relu'))  # 具有128个神经元和ReLU激活函数的隐藏层
model.add(Dense(10, activation='softmax'))  # 具有10个神经元（用于10个类别）和softmax激活函数的输出层

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 在测试集上评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'测试准确率：{test_acc}')
