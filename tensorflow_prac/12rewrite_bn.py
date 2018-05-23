# -*- coding:utf-8 -*-


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# 收敛：迭代算法求到解之后（数值解），在某个解上面迭代不动了，只会在可能的真实解两边荡秋千，显示出来目标函数值也是一样，这时就认为收敛了。

ACTIVATION = tf.nn.tanh    # 激活函数
N_LAYERS = 7               # 网络层数
N_HIDDEN_UNITS = 30        # 隐藏层神经元个数


def fix_seed(seed=1):
    # reproducible
    np.random.seed(seed)
    tf.set_random_seed(seed)


def plot_his(inputs, inputs_norm):
    # plot histogram for the inputs of every layer
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j*len(all_inputs)+(i+1))
            # 图表的整个绘图区域被分成2行和len(all_inputs)列, 现在画第j*len(all_inputs)+(i+1)个图
            plt.cla()    # plt.cla()清除轴, 当前活动轴在当前图中. 它保持其他轴不变
            if i == 0:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            # ravel行优先展开成一维, bins直方图数量, range每个柱状图取值范围(由于第一个图较离散, 所以取值方位较大)
            # 直方图,
            plt.yticks(())    # 设置y轴显示坐标为空
            if j == 1:
                plt.xticks(the_range)    # 设置x轴显示的坐标
            else:
                plt.xticks(())    # 设置x轴显示的坐标
            ax = plt.gca()    # 返回当前axes对象的句柄值
            # 把右边和上边的边界设置为不可见
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
    plt.draw()
    plt.pause(0.01)


def built_net(xs, ys, norm):
    def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
        # weights and biases (bad initialization for this case)
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

        # fully connected product
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        # normalize fully connected product
        if norm:
            # Batch Normalize
            # [1, 30], [1, 30]
            fc_mean, fc_var = tf.nn.moments(   # 计算统计矩, fc_mean是一阶矩即均值，fc_var则是二阶中心矩即方差, axes=[0]表示按列计算(干掉行)
                Wx_plus_b,    # [2500, 30]
                axes=[0],    # 想要 normalize 的维度, [0] 代表 batch 维度
                # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
            )
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001

            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.9999)    # 指定decay参数创建实例, 为了使模型趋于收敛, 会选择decay为接近1的数,
                                                                     # decay越大模型越稳定, 因为decay越大, 参数更新的速度就越慢，趋于稳定
            # 意义: 如果你是使用 batch 进行每次的更新, 那每个 batch 的 mean/var 都会不同, 所以我们可以使用 moving average
            #       的方法记录并慢慢改进 mean/var 的值. 然后将修改提升后的 mean/var 放入 tf.nn.batch_normalization()
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])    # 对模型变量使用apply方法
                # apply方法会为每个变量（也可以指定特定变量）创建各自的shadow variable, 即影子变量. 之所以叫影子变量,
                # 是因为它会全程跟随训练中的模型变量. 影子变量会被初始化为模型变量的值, 然后, 每训练一个step, 就更新一次.
                # 更新的方式为:shadow_variable = decay * shadow_variable + (1 - decay) * updated_model_variable
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
                # 对于control_dependencies这个管理器, 只有当里面的操作是一个op时, 才会生效, 也就是先执行传入的参数op,
                # 再执行里面的op. 例如而y=x仅仅是tensor的一个简单赋值, 不是定义的op, 所以在图中不会形成一个节点,
                # 这样该管理器就失效了. tf.identity是返回一个一模一样新的tensor的op, 这会增加一个新节点到gragh中,
                # 这时control_dependencies就会生效, 所以第二种情况的输出符合预期.
            mean, var = mean_var_with_update()    # both [1, 30]

            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
            # similar with this two steps:
            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)   主要做normalize的步骤
            # Wx_plus_b = Wx_plus_b * scale + shift   扩大(scale)和平移(shift), 可以被训练

        # activation
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        return outputs

    fix_seed(1)

    if norm:
        # BN for the first input
        fc_mean, fc_var = tf.nn.moments(
            xs,
            axes=[0],
        )
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001
        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)
        mean, var = mean_var_with_update()
        xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)

    # record inputs for every layer
    layers_inputs = [xs]

    # build hidden layers
    for l_n in range(N_LAYERS):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value     # get_shape()[1]第一次和后6次不一样, 第一次是1, 后六次是从网络出来的结果, 有30个神经元, 所以为30
                                                              # 下标0的值为?原因是xs定义为[None, 1], 所以0下标始终为None
        output = add_layer(
            layer_input,    # input
            in_size,        # input size
            N_HIDDEN_UNITS, # output size
            ACTIVATION,     # activation function
            norm,           # normalize before activation
        )
        layers_inputs.append(output)    # add output for next run

    # build output layer
    prediction = add_layer(layers_inputs[-1], 30, 1, activation_function=None)

    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))    # [2500, 1]
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op, cost, layers_inputs]

# make up data
fix_seed(1)
x_data = np.linspace(-7, 10, 2500)[:, np.newaxis]
np.random.shuffle(x_data)    # This function only shuffles the array along the first axis of a multi-dimensional array.
                             #  The order of sub-arrays is changed but their contents remains the same.
noise = np.random.normal(0, 8, x_data.shape)
y_data = np.square(x_data) - 5 + noise


# plot input data
# plt.scatter(x_data, y_data)
# plt.show()

xs = tf.placeholder(tf.float32, [None, 1])  # [num_samples, num_features]
ys = tf.placeholder(tf.float32, [None, 1])


train_op, cost, layers_inputs = built_net(xs, ys, norm=False)   # without BN
train_op_norm, cost_norm, layers_inputs_norm = built_net(xs, ys, norm=True) # with BN

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# record cost
cost_his = []
cost_his_norm = []
record_step = 5

plt.ion()
plt.figure(figsize=(7, 3))

for i in range(250):
    if i % 50 == 0:
        # plot histogram
        all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm], feed_dict={xs: x_data, ys: y_data})
        plot_his(all_inputs, all_inputs_norm)

    # train on batch
    sess.run([train_op, train_op_norm], feed_dict={xs: x_data[i*10:i*10+10], ys: y_data[i*10:i*10+10]})
    # print(sess.run([mean, var], feed_dict={xs: x_data, ys: y_data}))
    if i % record_step == 0:
        # record cost
        cost_his.append(sess.run(cost, feed_dict={xs: x_data, ys: y_data}))
        cost_his_norm.append(sess.run(cost_norm, feed_dict={xs: x_data, ys: y_data}))

plt.ioff()
plt.figure()
print(len(np.arange(len(cost_his))*record_step))
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his), label='no BN')     # no norm
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his_norm), label='BN')   # norm
plt.legend()   # 添加图例label
plt.show()
