# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/5/22 9:42
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 设置随机种子
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

# 参数设置
INPUT_NODE = 784   # 输入层的节点数。784=28*28
OUTPUT_NODE = 10  # 输出层的节点数，即类别的数目，即0~9
BATCH_SIZE = 64  # 一个训练batch中的训练数据个数。

LEARNING_RATE_BASE = 0.25    # 学习率
LEARNING_RATE_DECAY = 0.99   # 学习率的衰减率
REGULARIZATION_RATE = 0.0001   # 正则化项在损失函数中的系数
TRAINING_STEPS = 30000   # 训练轮数
MOVING_AVERAGE_DECAY = 0.99   # 滑动平均衰减率，decay越大模型越趋于稳定


# 计算神经网络的前向传播结果
def forward_propagation(input, avg_class, weights1, biases1):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input, weights1) + biases1)
        return layer1
    else:
        layer1 = tf.nn.relu(tf.matmul(input, avg_class.average(weights1)) + avg_class.average(biases1))
        return layer1

# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 参数初始化
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, OUTPUT_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算神经网络前向传播的结果。用于计算滑动平均的类为None
    y = forward_propagation(x, None, weights1, biases1)    # validation: (5000, 10)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 使用滑动平均之后的输出值，在计算交叉熵时仍然使用y，在最后输出时使用average_y
    average_y = forward_propagation(x, variable_averages, weights1, biases1)
    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数,logits表示隐藏层线性变换后非归一化后的结果,label是神经网络的期望输出(其输入格式需要是(batch_size))，y_是稀疏表示的，需要转化为非系数表示
    # argmax()输出的是每一列最大值所在的数组下标
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失，而不是用偏置项
    regularization = regularizer(weights1)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,     # 基础学习率
        global_step,    # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,     # 扫苗完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY,     # 学习率衰减速度
        staircase=True
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 每过一遍数据通过反向传播来更新神经网络的参数和参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程：
    with tf.Session() as sess:
        tf.global_variables_initializer().run()    # 初始化参数
        validate_feed_dict = {
            x: mnist.validation.images,     # (5000, 784)
            y_: mnist.validation.labels     # (5000, 10)
        }

        test_feed_dict = {
            x: mnist.test.images,     # (10000, 784)
            y_: mnist.test.labels     # (10000, 10)
        }

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed_dict)
                print('After %s training steps, accuracy in validation data is %g' % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed_dict)
        print('After %s training steps, accuracy in test data is %g' % (TRAINING_STEPS, test_acc))

if __name__ == '__main__':
    # 下载MNIST数据
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    train(mnist)







