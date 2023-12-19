"""
使用 keras 实现深度神经网络训练，步骤
     1、构建模型，设定初始值
     2、构建训练数据集和测试数据集
     3、对数据进行判断
"""



from tensorflow.keras import models,layers
def modulefinder():
    #1、初始化模型
    model = models.Sequential()
    model.add(layers.Dense(256,activation="relu",input_shape=(28*28,)))

    model.add(layers.Dense(512,activation="relu",input_shape=(28*28,)))

    # 2、加入第一个隐藏层
    #model.add(layers.Dense(256,activation="relu",input_shape=(28*28,)))
    # 3、加入第二个隐藏层
    # 4、加入输出层
    model.add(layers.Dense(10, activation="softmax"))
    # 5、编译模型
    # optimizer 更新模型参数以最小化损失函数的算法
    # SGD（随机梯度下降）：通过计算每个样本的梯度来更新参数，适用于大规模数据集。
    # Adam：结合了动量法和自适应学习率的优化器，通常在深度学习中表现较好。
    # RMSprop：使用梯度平方的移动平均来调整学习率，适用于非平稳目标函数。
    # Adagrad：根据参数的历史梯度累积来调整学习率，适用于稀疏数据集。
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model


from tensorflow.keras import datasets,utils
def createDate():
    #1、获取数据集
    (train_images, train_labels), (test_images, test_labels)  = datasets.mnist.load_data()
    #2、对数据进行归一化
    train_imgs = train_images.reshape((60000, 28*28))
    test_imgs = test_images.reshape((10000, 28*28))
    train_imgs = train_imgs.astype('float32') / 255
    test_imgs = test_imgs.astype('float32') / 255
    # 3、对测试lab和训练lab进行 one-hot 编码
    train_labs = utils.to_categorical(train_labels)
    test_labs = utils.to_categorical(test_labels)
    return train_imgs,train_labs,test_imgs,test_labs

import matplotlib.pyplot as plt
def getPredictImg():
    (train_images, train_labels), (test_images, test_labels)  = datasets.mnist.load_data()
    predict_img = test_images[100]
    plt.imshow(predict_img)
    plt.show()
    predict_img = predict_img.reshape((1,28*28))
    return predict_img


if __name__ == '__main__':
    #1、构建模型
    model = modulefinder()
    model.summary()
    #2、构建训练集和测试集
    train_img,train_lab,test_img,test_lab = createDate()
    #3、训练模型
    model.fit(train_img,train_lab,epochs=5,batch_size=128)
    #4、评估模型
    test_loss, test_acc = model.evaluate(test_img, test_lab)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)
    #5、预测值
    predict_img = getPredictImg()
    result = model.predict(predict_img)
    for i in range(result.shape[1]):
        if (result[0][i] == 1):
            print("the number for the picture is : ", i)
            break




