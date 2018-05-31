# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)    # 图级seed, 不同的sess间使用同一随机random变量产生同时随机相同

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001    # 学习率
training_iters = 100000    # 学习图片总数
batch_size = 128    # 一批学习图片数

n_inputs = 28   # MNIST data input (img shape: 28*28), 循环神经网络, 每行的28个像素点共同作为一次练习的输入
n_steps = 28    # time steps, 循环神经网络, 要一行一行练习, 此变量为一张图片的练习次数, 即像素行数
n_hidden_units = 120   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 120)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (120, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (120, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),    # [n_hidden_units个数, 只有一个维度, +每行都加一次]
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):     # https://zhuanlan.zhihu.com/p/28919765
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])    # 128 batch * 28 steps, n_inputs一次输入一行的28个像素

    # into hidden
    # X_in = (128 batch * 28 steps, 120 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 120 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])


    # cell
    ##########################################

    # basic LSTM Cell.
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)        # 定义cell有多少个神经元
    # lstm cell is divided into two attributes (c_state, h_state)    c_state:主线剧情  h_state:分线剧情
    global init_state
    init_state = cell.zero_state(batch_size, dtype=tf.float32)        # 通过zero_state得到一个全0的初始状态, 形状为(batch_size, state_size)

    global final_state
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    # 使用dynamic_rnn函数就相当于调用了n次call函数。即通过{h0,x1, x2, …., xn}直接得{h1,h2…,hn}
    # time_major: The shape format of the inputs and outputs Tensors. If true, these Tensors must be shaped [max_time, batch_size, depth]
    # If false, these Tensors must be shaped [batch_size, max_time, depth]
    # outputs就是练习28次每次一次的结果
    # final_state是最后的(c状态, h状态), c就是和h并列的另一个循环的部件, 可以理解成LSTM隐层由两部分组成, 一部分叫h, 另一部分叫c就行了

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']
    # https://zhuanlan.zhihu.com/p/28919765
    # # or
    # unpack to list [(batch, outputs)..] * steps
    global outputs2
    outputs2 = tf.unstack(tf.transpose(outputs, [1, 0, 2]))    # old 1
    # transpose交换前两个维度变为[训练次数, 一个批次个数, 神经元个数]:[28,128,120], unstack默认axis为0, 即将第一个维度拆开成28个
    results = tf.matmul(outputs2[-1], weights['out']) + biases['out']    # shape = (120, 10)
    # new function:
    # results = tf.layers.dense(outputs[:, -1, :], 10)              # output based on the last output step

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))    # y:[120, 10]    old2
# new function:
# cost = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)       # compute cost


train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))    # old 3
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# new functions:
# accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
#     labels=tf.argmax(y, axis=1), predictions=tf.argmax(pred, axis=1),)[1]    # axis=1 在第二个维度找最大

with tf.Session() as sess:
    init = tf.global_variables_initializer()    # old3
    sess.run(init)
    # new function:
    # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())  # the local var is for update_op
    # sess.run(init)  # initialize var in graph
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_op, feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1

        print(sess.run(tf.shape(final_state[1]), feed_dict={x: batch_xs,  y: batch_ys}))
        # f_state = sess.run(final_state, feed_dict={x: batch_xs,  y: batch_ys})
        # out = sess.run(outputs2,  feed_dict={x: batch_xs,  y: batch_ys})
        # print(f_state[0]==out[0])
