# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np


# create data

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3


# ######### create tensorflow structure start #########

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))    # 一维, -1.0~1.0, 权重值
biases = tf.Variable(tf.zeros([1]))    # 一维, 全0, 偏移值

y = Weights*x_data + biases    # 待训练

loss = tf.reduce_mean(tf.square(y - y_data))    # 求出误差值方差
optimizer = tf.train.GradientDescentOptimizer(0.5)    # 使用梯度下降法, 学习率0.5
train = optimizer.minimize(loss)    # 使用梯度下降法根据误差不断优化训练

# ######### create tensorflow structure end #########



sess = tf.Session()    # 执行训练对象
init = tf.global_variables_initializer()    # 初始化数据方法

sess.run(init)    # 执行初始化数据方法
for step in range(201):
    sess.run(train)    # 开始训练
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

