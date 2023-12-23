import tensorflow as tf

state = tf.Variable(0, name="counter")

one = tf.constant(1)

new_val = tf.add(state, one)
update = tf.assign(state, new_val)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 运行 'init' op
    sess.run(init_op)
    # 打印 'state' 的初始值
    print("state", sess.run(state))

    # 运行 op, 更新 'state', 并打印 'state'
    for _ in range(5):
        print("update", sess.run(update))
        print("state", sess.run(state))