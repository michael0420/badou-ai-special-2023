import tensorflow as tf

input1 = tf.constant(1)
input2 = tf.constant(2)
input3 = tf.constant(3)

add = tf.add(input1, input2)
mul = tf.multiply(input3, input2)

with tf.Session() as sess:
    result = sess.run([add, mul])

    print(result)


