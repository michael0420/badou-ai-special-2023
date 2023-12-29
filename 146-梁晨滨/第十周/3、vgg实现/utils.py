import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf


# 读取图片并裁剪成正方形
def load_image(path):

    img = mpimg.imread(path)
    short_edge = min(img.shape[:2])

    x0 = int((img.shape[1] - short_edge) / 2)
    y0 = int((img.shape[0] - short_edge) / 2)
    x1, y1 = x0 + short_edge, y0 + short_edge
    crop_img = img[y0: y1, x0: x1]

    return crop_img


# 图片整形，(H, W, C) -> (1, H, W, C) (对齐角点选项)
def resize_image(image, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size, method, align_corners)
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
        return image


# 打印预测结果top1和top5的, 最终预测结果是top1的
def print_predict(prob, file_path):

    synset = [l.strip() for l in open(file_path).readlines()]
    pred = np.argsort(prob)[::-1]

    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1



