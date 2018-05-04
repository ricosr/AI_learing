# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)    # mnist数据集


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_b = tf.matmul(inputs, Weights) + biases
    Wx_b = tf.nn.dropout(Wx_b, keep_prob)    # 防止过拟合的方法, 一直没成功过。。。
    if activation_function:
        return activation_function(Wx_b)
    else:
        return Wx_b


def calculate_accuracy(tx, ty):    # 输入为test数据, 用于测试准确度
    global prediction
    y_p = sess.run(prediction, feed_dict={xs: tx, keep_prob:1.0})    # 拿到当前网络状态下(当前的权值和偏移)的预测值
    correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(ty, 1))    # 对比预测值和test真实值1的位置
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    # cast将bool值映射成float值
    result = sess.run(accuracy, feed_dict={xs: tx, ys: ty})
    tf.summary.histogram("result", result)
    return result


keep_prob = tf.placeholder(tf.float32)    # 保留权重值对拟合结果的影响百分比
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, tf.nn.softmax)    # softmax回归模型
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)))    # 交叉熵用来衡量我们的预测用于描述真相的低效性
tf.summary.scalar("loss", cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("./log/train", sess.graph)
test_writer = tf.summary.FileWriter("./log/test", sess.graph)k
init = tf.global_variables_initializer()

sess.run(init)


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})
    if i % 50 ==0:
        print(calculate_accuracy(mnist.test.images, mnist.test.labels))
        # print(sess.run(tf.shape(Wx_b), feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1}))
        p = (sess.run(prediction, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1}))
        print(sess.run(tf.argmax(prediction, 1), feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1}))
        print(p[-1])
        # print(sess.run(tf.argmax(p[0], 1)))

        print("****************************************\n")
        data_train = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1.0})
        data_test = sess.run(merged, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0})
        train_writer.add_summary(data_train, i)
        test_writer.add_summary(data_test, i)
