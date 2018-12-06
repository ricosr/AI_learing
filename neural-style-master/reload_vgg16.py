# -*- encoding:utf-8 -*-

import tensorflow as tf


class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy=None, step=None, shape=None, new_img=None):
        self.data_dict = vgg16_npy
        if step == "optimize":
            self.tfx = new_img
        else:
            self.tfx = tf.placeholder(tf.float32, shape=shape)

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(pool3, "conv4_1")
        if step == "style" or "optimize":
            conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            conv4_3 = self.conv_layer(conv4_2, "conv4_3")
            pool4 = self.max_pool(conv4_3, 'pool4')
            self.conv5_1 = self.conv_layer(pool4, "conv5_1")
        if step == "optimize":
            self.layers_ls = []
            self.layers_ls.append(self.conv1_1)
            self.layers_ls.append(self.conv2_1)
            self.layers_ls.append(self.conv3_1)
            self.layers_ls.append(self.conv4_1)
            self.layers_ls.append(self.conv5_1)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def train(self, step, input_x):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if step == "content":
            result = self.sess.run(self.conv3_1, {self.tfx: input_x})
        if step == "style":
            result = self.sess.run([self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1, self.conv5_1], {self.tfx: input_x})
        return result
