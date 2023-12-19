from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
(train_img,train_lab),(test_img,test_lab) = mnist.load_data()

nn = models.Sequential()
nn.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
nn.add(layers.Dense(10,activation='softmax'))

nn.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

train_img = train_img.reshape(60000,(28*28))
train_img = train_img.astype('float')/255
test_img = test_img.reshape(10000,(28*28))
test_img = test_img.astype('float')/255

train_labels = to_categorical(train_lab)
test_labels = to_categorical(test_lab)


nn.fit(train_img, train_labels, epochs=5, batch_size=128)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

test_images = test_images.reshape((10000, 28 * 28))
res = nn.predict(test_img)
#注意区分 print(res.shape[0])和print(res[0].shape),前者输出10000，后者输出10


i = np.argmax(res[1])
print(i)


'''
import numpy as np  
  
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  
print(np.argmax(a))  # 输出：8  
  
print(np.argmax(a, axis=0))  # 输出：[2 1 0]  
print(np.argmax(a, axis=1))  # 输出：[2 2 2]
'''

# for i in range(res[1].shape[0]):
#
#     if (res[1][i] == 1):
#         print("the number for the picture is : ", i)
#         break