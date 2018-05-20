# -*- coding:utf-8 -*-


# 一批50组样本，一组样本练习20个

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

BATCH_START = 0    # x轴开始坐标
TIME_STEPS = 20    # cell一组样本一次要学习的次数
BATCH_SIZE = 50    # 一个batch的训练样本数
INPUT_SIZE = 1     # 单次只能输入一个样本
OUTPUT_SIZE = 1    # 单次只返回一个结果
CELL_SIZE = 10     # cell隐藏层有10个神经元
LR = 0.006         # 学习率


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    # (10*np.pi)无所谓就是让数据密度足够大提高准确性和画图好看一点
    # TIME_STEPS*BATCH_SIZE为了reshape时数据个数足够展开
    seq = np.sin(xs)    # 只是作为训练的初始值, 这样对于cos来说容易拟合, 省略了激励函数, 如果不加sin, 搜索 **activate** 处代码可激活
    res = np.cos(xs)
    BATCH_START += TIME_STEPS    # 将x轴坐标每次向后移动20个单位
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps    # 学习次数
        self.input_size = input_size    # 一次学习输入数据维度大小
        self.output_size = output_size    # 输出维度
        self.cell_size = cell_size    # 一个cell包含神经元个数
        self.batch_size = batch_size    # 一个batch训练数据个数
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')    # (batch, n_step, in_size)
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='3_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in    # 确保每个神经元都对所有数据进行干预训练，tf.matmul(l_in_x, Ws_in):
         #                        data                                  weights                                        cell_size的排列没变
         #             in_size(1) ... in_size(n)           cell_size(1) ... cell_size(n)                    cell_size(1) ... cell_size(n)
         # batch*n_step                             in_size                                     batch*n_step
         # ...                                   *  ...                                     =   ...
         # ...                                      ...                                         ...
         # batch*n_step                             in_size                                     batch*n_step
        # reshape l_in_y ==> (batch, n_steps, cell_size)    # 最后一个维度实际上是训练样本数据乘权值加偏移后的值
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
        # self.l_in_y = tf.cos(tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D'))    # **activate**

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)    # state_is_tuple会保存长时记忆，返回(outputs, final_state)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        # 定义交叉熵损失函数, TensorFlow提供了sequence_loss_by_example函数来计算一个序列的交叉熵的和,一个序列的......
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],     # 预测的结果, 这里将[batch * steps, output_size]二维数组压缩成一维数组
            [tf.reshape(self.ys, [-1], name='reshape_target')],     # 期待的正确答案, 这里将[batch * steps, output_size]二维数组压缩成一维数组
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],    # 损失的权重, 在这里所有的权重都为1，也就是说不同batch和不同时刻的重要程度是一样的
            average_across_timesteps=True,    # If set, divide the returned cost by the total label weight.
            softmax_loss_function=self.ms_error,    # 计算误差的函数
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(tf.reduce_sum(losses, name='losses_sum'), self.batch_size, name='average_cost')
            # div是除法, 浮点数除以浮点数得浮点数, 否则为整数
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))    # 误差平方和

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    plt.ion()
    plt.show()
    for i in range(200):
        seq, res, xs = get_batch()
        if i == 0:
            feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state, pred, in_y = sess.run([model.train_op, model.cost, model.cell_final_state, model.pred, tf.shape(model.pred)], feed_dict=feed_dict)
        # print(in_y)
        # print(pred)
        # print(res.shape)
        # print(res[0])
        # print(res)
        #
        # time.sleep(10)

        # plotting
        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
        # xs[0, :], res[0].flatten() 取第0行的x和对应的cos(x)的点的坐标，一次20个
        # 因为output layer最后返回的结果是(batch * steps, output_size), output_size为1, 所以截取前TIME_STEPS个, 也就是20
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)

        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
