import numpy as np
import matplotlib.pyplot as plt

file = open("dataset/mnist_train.csv")
train_data = file.readlines()
file.close()

data_0 = train_data[0].split(',')
picture_num = data_0[0]
picture_array = np.asfarray(data_0[1:]).reshape(28, 28)

plt.imshow(picture_array, cmap='Greys')
plt.axis('off')
plt.show()

