---
layout:     post
title:      "TensorFlow学习笔记（6）--循环神经网络"
date:       2017-12-21 19:00:00
author:     "SH"
header-img: "img/post_bg_headset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - TensorFlow
---



## 1.简介

​	循环神经网络主要用于处理和预测序列数据，会记忆之前的信息，并利用之前的信息影响后面节点的输出。

​	循环神经网络要求每一个时刻都有一个输入，但是不一定每个时刻都需要输出。总损失为所有时刻（或部分时刻）上的损失函数的总和。

​	理论上循环神经网络可以支持任意长度的序列，然而在实际中，如果序列过长会导致优化是出现梯度消散的问题，所以一般会规定一个最大长度，当序列长度超过规定长度后会对序列进行截断。

​	循环神经网络的前向传播过程：
```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-20 18:31
# @Author  : Storm
# @File    : chapter08-01.py
# 循环神经网络前向传播过程

import numpy as np
from tensorflow.python.ops import rnn_cell

X = [1, 2]
state = [0.0, 0.0]

# 分开定义不同输入部分的权重以方便操作
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

# 定义用于输出的全连接层参数
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 按照时间顺序执行循环神经网络的前向传播过程。
for i in range(len(X)):
    # 计算循环体中的全连接层神经网络。
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)

    # 根据当前时刻状态计算最终输出。
    final_output = np.dot(state, w_output) + b_output

    # 输出每个时刻的信息
    print("before activation:", before_activation)
    print("state:", state)
    print("output:", final_output)

```



## 2.长短时记忆网络（LSTM）

​	LSTM靠一些“门”的结构让信息有选择性的影响循环神经网络中每个时刻的状态。所谓“门”的结构就是一个使用sigmoid神经网络和一个按位做乘法的操作。

​	“遗忘门”的作用是让循环神经网络“忘记”之前没有用的信息。“输入门”用来从当前的输入补充最新的记忆。

​	双向循环神经网络（bidirectional RNN）：由两个循环神经网络上下叠加在一起组成的。

​	深层循环神经网络（deepRNN）：将每一个时刻上的循环体重复多次。TensorFlow提供了MultiRNNCell类来实现深层循环神经网络的前向传播过程。

​	循环神经网络的dropout：一般只在不同层循环体结构之间使用dropout，而不在同一层的循环体结构之间使用。TensorFlow使用DropoutWrapper类可以很容易实现dropout功能。



## 3.PTB数据集

​	PTB（Penn Treebank dataset）文本数据集是语言模型学习中使用很多的数据集。下载地址：http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

​	TensorFlow对PTB数据集提供了支持：
```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-20 22:20
# @Author  : Storm
# @File    : chapter08-02.py
# PTB数据集

import tensorflow as tf
import reader

DATA_PATH = "datasets/PTB_data"

# 1. 读取数据并打印长度及前100位数据。
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
print(len(train_data))
print(train_data[:100])

# 2. 将训练数据组织成batch大小为4、截断长度为5的数据组。并使用队列读取前3个batch。
# ptb_producer返回的为一个二维的tuple数据。
result = reader.ptb_producer(train_data, 4, 5)

# 通过队列依次读取batch。
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        x, y = sess.run(result)
        print("X%d: " % i, x)
        print("Y%d: " % i, y)
    coord.request_stop()
    coord.join(threads)

```



## 4.使用循环神经网络实现语言模型

```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-21 10:23
# @Author  : Storm
# @File    : chapter08-03.py
# 使用循环神经网络实现语言模型

import numpy as np
import tensorflow as tf
import reader

DATA_PATH = "datasets/PTB_data"

HIDDEN_SIZE = 200  # 隐藏层规模
NUM_LAYERS = 2  # 深层循环神经网络中LSTM结构的层数
VOCAB_SIZE = 10000  # 词典规模，加上语句结束标识符和稀有单词标识符总共一万个单词

LEARNING_RATE = 1.0  # 学习速率
TRAIN_BATCH_SIZE = 20  # 训练数据batch的大小
TRAIN_NUM_STEP = 35  # 训练数据截断长度

# 在测试时不需要使用截断，所以可以将测试数据看成一个超长的序列。
EVAL_BATCH_SIZE = 1  # 测试数据batch的大小
EVAL_NUM_STEP = 1  # 测试数据截断长度
NUM_EPOCH = 2  # 使用训练数据的轮数
KEEP_PROB = 0.5  # 节点不被dropout的概率
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的参数


# 通过一个类来描述模型结构，这样方便维护循环神经网络中的状态。
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用的batch大小和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义输入层。
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义使用LSTM结构及训练时使用dropout。
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        # 初始化最初的状态。
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将原本单词ID转为单词向量。
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # 定义输出列表。
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        # 定义交叉熵损失函数和平均损失。
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        # 只在训练模型时定义反向传播操作。
        if not is_training: return
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤。
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


# 使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity值
def run_epoch(session, model, data, train_op, output_log, epoch_size):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 训练一个epoch。
    for step in range(epoch_size):
        x, y = session.run(data)
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                     {model.input_data: x, model.targets: y, model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        if output_log and step % 10 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))
    return np.exp(total_costs / iters)


def main():
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    # 计算一个epoch需要训练的次数
    train_data_len = len(train_data)
    train_batch_len = train_data_len // TRAIN_BATCH_SIZE
    train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP

    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len // EVAL_BATCH_SIZE
    valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP

    test_data_len = len(test_data)
    test_batch_len = test_data_len // EVAL_BATCH_SIZE
    test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_queue = reader.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)
        eval_queue = reader.ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps)
        test_queue = reader.ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            run_epoch(session, train_model, train_queue, train_model.train_op, True, train_epoch_size)

            valid_perplexity = run_epoch(session, eval_model, eval_queue, tf.no_op(), False, valid_epoch_size)
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, eval_model, test_queue, tf.no_op(), False, test_epoch_size)
        print("Test Perplexity: %.3f" % test_perplexity)

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()

```



## 5.TFLearn自定义模型

​	TFLearn对训练TensorFlow模型进行了一些封装，使其更便于使用。

​	使用TFLearn对iris数据集进行分类：
```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-21 11:06
# @Author  : Storm
# @File    : chapter08-04.py
# TFLearn自定义模型

from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat

learn = tf.contrib.learn


# 自定义softmax回归模型。
def my_model(features, target):
    target = tf.one_hot(target, 3, 1, 0)

    # 计算预测值及损失函数。
    logits = tf.contrib.layers.fully_connected(features, 3, tf.nn.softmax)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    # 创建优化步骤。
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=0.01)
    return tf.arg_max(logits, 1), loss, train_op


# 读取数据并将数据转化成TensorFlow要求的float32格式。
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0)

x_train, x_test = map(np.float32, [x_train, x_test])

# 封装和训练模型，输出准确率。
classifier = SKCompat(learn.Estimator(model_fn=my_model, model_dir="datasets/Models/model_1"))
classifier.fit(x_train, y_train, steps=800)

y_predicted = [i for i in classifier.predict(x_test)]
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: %.2f%%' % (score * 100))
```



## 6.预测正弦函数

​	代码报错！！！ ？？？

```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-21 11:12
# @Author  : Storm
# @File    : chapter08-05.py
# 预测正弦函数

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from tensorflow.python.ops import array_ops as array_ops_
import matplotlib.pyplot as plt

learn = tf.contrib.learn

# 1.设置神经网络的参数。
HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIMESTEPS = 10
TRAINING_STEPS = 3000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01


# 2.定义生成正弦数据的函数。
def generate_data(seq):
    X = []
    y = []

    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# 3.定义lstm模型。
def lstm_model(X, y):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)

    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(output, [-1, HIDDEN_SIZE])
    # output = tf.reshape(tf.concat(output, 1), [-1, HIDDEN_SIZE])

    # 通过无激活函数的全联接层计算线性回归，并将数据压缩成一维数组的结构。
    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    # 将predictions和labels调整统一的shape
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])

    loss = tf.losses.mean_squared_error(predictions, labels)

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)

    return predictions, loss, train_op


# 4.进行训练。
# 封装之前定义的lstm。
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir="datasets/Models/model_2"))

# 生成数据。
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))

# 拟合数据。
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

# 计算预测值。
predicted = [[pred] for pred in regressor.predict(test_X)]

# 计算MSE。
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print("Mean Square Error is: %f" % rmse[0])

# 5. 画出预测值和真实值的曲线。
plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
plt.show()
plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
plt.show()

```


TensorFlow 实战Google深度学习框架（第八章）