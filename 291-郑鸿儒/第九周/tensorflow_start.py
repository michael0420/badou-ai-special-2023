#!/usr/bin/env python
# encoding=utf-8
import tensorflow as tf


# state = tf.Variable(0, name="counter")
# one = tf.constant(1)
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)
#
# initial = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(initial)
#     # sess.run(update)
#     print(sess.run(state))
#     for i in range(5):
#         sess.run(update)
#     print(sess.run(state), state.name)
#     print(tf.version.VERSION)

# matrix_1 = tf.constant([[3, 3]])
# matrix_2 = tf.constant([[2], [2]])
#
# product = tf.matmul(matrix_1, matrix_2)
#
# with tf.Session() as sess:
#     res = sess.run(product)
# print(res)

# input_1 = tf.constant(3)
# input_2 = tf.constant(4)
# input_3 = tf.constant(5)
#
# intermd = tf.add(input_2, input_3)
# multi = tf.multiply(input_1, intermd)
#
# with tf.Session() as sess:
#     res = sess.run([intermd, multi])
#     print(sess.run(input_1))
# print(res)

# input_1 = tf.placeholder(tf.float32)
# input_2 = tf.placeholder(tf.float32)
#
# product = tf.multiply(input_1, input_2)
#
# with tf.Session() as sess:
#     first = input("输入数字1，回车结束")
#     second = input("输入数字2，回车结束")
#     res = sess.run(product, feed_dict={input_1: first, input_2: second})
#
# print(res)
