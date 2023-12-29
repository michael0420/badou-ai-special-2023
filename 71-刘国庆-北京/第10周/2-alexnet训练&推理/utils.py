# 导入OpenCV库，用于计算机视觉任务
import cv2
# 导入Matplotlib库中的image模块，并将其重命名为mpimg
import matplotlib.image as mpimg
# 导入NumPy库，用于科学计算，重命名为np
import numpy as np
# 导入TensorFlow库，用于机器学习和深度学习，重命名为tf
import tensorflow as tf


# 函数的目的是加载一张图片并将其裁剪成中心的正方形
def load_image(path):
    # 读取图片，使用Matplotlib的image模块的imread函数
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    # 获取图片宽和高的最小值
    short_edge = min(img.shape[:2])
    # 计算垂直方向上的裁剪起始位置
    yy = int((img.shape[0] - short_edge) / 2)
    # 计算水平方向上的裁剪起始位置
    xx = int((img.shape[1] - short_edge) / 2)
    # 对图片进行裁剪，保留中心正方形区域
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # 返回正方形图片
    return crop_img


# 使用TensorFlow的name_scope为操作添加名称空间
def resize_image(image, size):
    # 创建一个空列表用于存储调整大小后的图片
    images = []
    # 遍历输入的图片列表
    for i in image:
        # 使用OpenCV的resize函数调整图片大小
        i = cv2.resize(i, size)
        # 将调整大小后的图片添加到列表中
        images.append(i)
    # 将图片列表转换为NumPy数组
    images = np.array(images)
    # 返回调整大小后的图片数组
    return images


# 定义打印答案的函数，输入参数为argmax
def print_answer(argmax):
    # 打开包含索引与对应词语的文件
    with open("./data/model/index_word.txt", "r", encoding='utf-8') as f:
        # 从文件中读取每一行，提取出词语部分，存储在synset列表中
        synset = [l.split(";")[1][:-1] for l in f.readlines()]
    # 打印索引对应的词语
    print(synset[argmax])
    # 返回索引对应的词语
    return synset[argmax]
