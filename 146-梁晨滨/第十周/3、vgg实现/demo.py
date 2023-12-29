from vgg16net import vgg_16 as vgg16
import tensorflow as tf
from utils import load_image, resize_image, print_predict


# 读取图片
img = load_image("../dog.jpg/dog.jpg")

# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32, [None, None, 3])
resized_img = resize_image(inputs, (224, 224))

# 建立网络结构
prediction = vgg16(resized_img)

# 载入模型
sess = tf.Session()
ckpt_filename = 'vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

# 最后结果进行softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs: img})

# 打印预测结果
print("result: ")
print_predict(pre[0], './synset.txt')
