import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w1 = tf.Variable(tf.random.normal([1, 10]))
        self.b1 = tf.Variable(tf.zeros([1, 10]))
        self.w2 = tf.Variable(tf.random.normal([10, 1]))
        self.b2 = tf.Variable(tf.zeros([1, 1]))

    # 进行前向计算
    def call(self, inputs):
        # 输入层到中间层
        r1 = tf.matmul(inputs, self.w1) + self.b1
        l1 = tf.nn.tanh(r1)  # (200, 10)

        # 中间层到输出层
        r2 = tf.matmul(l1, self.w2) + self.b2
        return tf.nn.tanh(r2)


# 加载MNIST数据集
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 使用numpy生成200个随机点
x_train = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_train.shape)
y_train = np.square(x_train) + noise
print('x_train', x_train.shape)
print('y_train', y_train.shape)

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

model = MyModel()

# 训练
for epoch in range(2000):


    # 定义优化器
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    with tf.GradientTape() as tape:
        # 损失函数(均方差,MSE)计算loss
        prediction = model(x_train)
        loss = tf.reduce_mean(tf.square(prediction - y_train))
        # print('Loss :{}'.format(loss))
        # print(model.trainable_variables)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 推理
prediction_value = model(x_train)

# 画图
plt.figure()
plt.scatter(x_train, y_train)
plt.plot(x_train, prediction_value, 'r-', lw=5)

plt.show()
