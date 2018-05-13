# -*- coding:utf-8 -*-
# test matmul and how to use + function


import tensorflow as tf
import numpy as np

x_data = np.linspace(-1, 1, 5)[:, np.newaxis].astype(np.float32)
Weights = tf.Variable(tf.random_normal([1, 3]), dtype=tf.float32)
biases = tf.Variable(tf.zeros([1, 3]) + 0.1, dtype=tf.float32)
# biases = tf.Variable(tf.constant(0.1, shape=[3, ]))


Wx_plus_b = tf.matmul(x_data, Weights) + biases   # 每行都加一次

ac = tf.nn.relu(Wx_plus_b)
initial = tf.constant(0.1, shape=[5])

test_bias = tf.Variable(tf.constant(0.1, shape=[2, ]))


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# x = sess.run(x_data)
print(x_data)
print()

w = sess.run(Weights)
print(w)
print()

b = sess.run(biases)
print(b)
print()

result = sess.run(Wx_plus_b)
print(result)
print("*********************")

a = sess.run(ac)
print(a)


print(sess.run(initial))

print(sess.run(test_bias))

