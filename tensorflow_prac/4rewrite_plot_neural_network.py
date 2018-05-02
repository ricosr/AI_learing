# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 定义一个神经层
def add_layer(inputs, input_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([input_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus = tf.matmul(inputs, Weights) + biases
    if activation_function:
        result = activation_function(Wx_plus)
    else:
        result = Wx_plus
    return result


# 定义样本数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


# define placeholder for x and y
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# 开始定义神经层
# 隐藏层
ly1 = add_layer(x_data, 1, 10, tf.nn.relu)    # 300个训练样本, 1个特征, 10个神经元
# 输出层
prediction = add_layer(ly1, 10, 1)    # 隐藏层10个神经元, 1个输出神经元


# 定义训练方法
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))    # 求方差, reduction_indices=[1]为干掉第二维度
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)    # 梯度下降法, 学习率为0.1


# 定义session
sess = tf.Session()
# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)


# 画图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()


# 开始训练
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except:
            pass

        prediction_val = sess.run(prediction, feed_dict={xs:x_data, ys:y_data})
        lines = ax.plot(x_data, prediction_val, 'r-', lw=5)
        plt.pause(1)




