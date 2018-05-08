# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)    # mnist数据集


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)    # shape表示生成张量的维度, mean是均值, stddev是标准差, 产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
    # 这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):    # Convolutional layer 提取图像
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')    # SAME 扫描图片时会全扫描, 出界部分会补0, VALID出界则会舍弃


def max_pool_2x2(x):    # polling layer max polling 提取一片区域的最大值
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # ksize  池化窗口的大小  是一个四位向量  一般为【1, height, width, 1】 因为我们不想在batch和channels上做操作,所以这两个维度上设为1
    # 第一个参数为x的shape为[batch, height, width, channels], 第三个参数, 和卷积类似, 窗口在每一个维度上滑动的步长, 所以一般设为[1, stride, stride, 1]


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])   # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# -1的含义是：我们自己不用去管这个维度的大小, 为缺省值, reshape会自动根据其他维度计算, 但是我的这个列表中只能有一个-1, 原因很简单, 多个-1会造成多解的方程情况
# 其实这个-1地方reshape函数算出的数就是samples的多少  也就是我们导入例子的多少  也就是导入了多少张图像
# print(x_image.shape)  # [n_samples, 28,28,1]


## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32])
# 本层我们的卷积核patch的大小是5x5, 因为黑白图片channel是1所以输入是1, 输出是32个featuremap
# 32就是操作后输出图片的厚度DEPTH, 神经网络是通过增加图片厚度来总结图片特征的, 这里32是随便取的, 你取30也无所谓
# 由于扫描图片有叠加, 加之权重, 为加权叠加, 所以depth会增加
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)    # output size 14x14x32


## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64])    # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)     # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)    # output size 7x7x64


## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))