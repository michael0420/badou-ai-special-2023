from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import cv2
import keras.utils as np_utils
import tensorflow as tf
import matplotlib.image as mpimg

import os
import numpy as np

'''
1、一张原始图片被resize到(224,224,3); 
2、使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层， 输出的shape为(55,55,96); 
3、使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96) 
4、使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层， 输出的shape为(27,27,256); 
5、使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256); 
6、使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层， 输出的shape为(13,13,384); 
7、使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层， 输出的shape为(13,13,384); 
8、使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层， 输出的shape为(13,13,256); 
9、使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256); 
10、两个全连接层，最后输出为1000类


'''

def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    # keras 初始化模型
    model = Sequential()
    # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    # 所建模型后输出为48特征层
    model.add(Conv2D(filters=48,kernel_size=(11, 11),strides=(4, 4),padding='valid',input_shape=input_shape,activation='relu'))
    # 通过规范化的手段,将越来越偏的分布拉回到标准化的分布,使得激活函数的输入值落在激活函数对输入比较敏感的区域,从而使梯度变大,加快学习收敛速度,避免梯度消失的问题
    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2),padding='valid'))
    # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
    # 所建模型后输出为128特征层
    model.add(Conv2D(filters=128, kernel_size=(5, 5),strides=(1, 1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2),padding='valid'))
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(Conv2D(filters=192,kernel_size=(3, 3),strides=(1, 1),padding='same',activation='relu'))
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(Conv2D(filters=192,kernel_size=(3, 3),strides=(1, 1),padding='same',activation='relu') )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
    # 所建模型后输出为128特征层
    model.add(Conv2D(filters=128,kernel_size=(3, 3),strides=(1, 1),padding='same',activation='relu'))
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)； padding 是否填充 valid时，表示边缘不填充
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2),padding='valid'))
    # 两个全连接层，最后输出为1000类,这里改为2类
    # 缩减为1024
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    # 通过忽略一定数量的特征检测器
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    # 输出数
    model.add(Dense(output_shape, activation='softmax'))

    return model

def train_data_lable_w():
    photos = os.listdir("train/")
    with open("alexnet_dataset.txt", "w") as f:
        for photo in photos:
            name = photo.split(".")[0]
            if name == "cat":
                f.write(photo + ";0\n")
            elif name == "dog":
                f.write(photo + ";1\n")
    f.close()
    print("数据标注完成")

def train_data_lable_r():
    with open(r"alexnet_dataset.txt", "r") as f:
        lines = f.readlines()
    f.close()
    return lines

'''
每批次的数据集进来，组装成 数据样本和标注集，然后输出
'''
def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r"./train" + '/' + name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = resize_image(X_train, (224, 224))
        X_train = X_train.reshape(-1,224,224,3)
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 2)
        yield (X_train, Y_train)

def train_process():
    lines = train_data_lable_r()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    # 90%用于训练，10%用于估计。
    train_num = int(len(lines) * 0.9)
    valid_num = len(lines) - train_num
    model = AlexNet()
    # 3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
         'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',monitor='acc',save_weights_only=False,save_best_only=True,period=3
    )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',factor=0.5,patience=3,verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10,verbose=1
    )

    # 交叉熵
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])

    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_num, valid_num, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:train_num], batch_size),
                        steps_per_epoch=max(1, train_num // batch_size),
                        validation_data=generate_arrays_from_file(lines[train_num:], batch_size),
                        validation_steps=max(1, valid_num // batch_size),
                        epochs=20,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr, early_stopping])
    model.save_weights('alexnet_last_weight.h5')

def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images
def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img


result = {0 : '猫', 1 : '狗' }

def predict():
    model = AlexNet()
    model.load_weights("alexnet_last_weight.h5")
    img1 = load_image("test.jpeg")
    img_RGB1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img_nor1 = img_RGB1 / 255
    img2 = load_image("dog1.jpeg")
    img_RGB2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_nor2 = img_RGB2 / 255

    image1_expanded = np.expand_dims(img_nor1, axis=0)
    image2_expanded = np.expand_dims(img_nor2, axis=0)
    merged_images = np.concatenate([image1_expanded, image2_expanded], axis=0)
    img_resize = resize_image(merged_images, (224, 224))
    for i in range(img_resize.shape[0]):
        current_image = img_resize[i]
        current_imagecopy = np.expand_dims(current_image, axis=0)
        index = np.argmax(model.predict(current_imagecopy))
        print("图片{%d}:alexnet模型预测为：" % i, result.get(index))
        cv2.imshow("原始图片", current_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    train_data_lable_w()
    train_process()
    predict()





