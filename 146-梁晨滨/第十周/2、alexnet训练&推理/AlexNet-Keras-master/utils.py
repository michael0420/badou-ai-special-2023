import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf

def load_image(path):

    img = mpimg.imread(path)
    short_edge = min(img.shape[:2])

    x0 = int((img.shape[1] - short_edge) / 2)
    y0 = int((img.shape[0] - short_edge) / 2)
    x1, y1 = x0 + short_edge, y0 + short_edge
    crop_img = img[y0: y1, x0: x1]

    return crop_img


def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images

def print_answer(argmax):
    with open("./data/model/index_word.txt", "r", encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]
        
    print(synset[argmax])
    return synset[argmax]