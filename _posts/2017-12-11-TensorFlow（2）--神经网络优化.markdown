---
layout:     post
title:      "TensorFlow学习笔记（2）--神经网络优化"
date:       2017-12-11 16:00:00
author:     "SH"
header-img: "img/post_bg_headset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - TensorFlow
---



## 1.损失函数

​   交叉熵 H(p,q) 是不对称的, 刻画的是两个概率分布之间的距离，通过概率分布q来表达概率分布p的困难程度，交叉熵值越小，两个概率分布越接近。

​   Softmax将神经网络的输出变成一个概率分布。

​   交叉熵一般会与Softmax回归一起使用，TensorFlow对这两个功能进行了统一封装，提供了tf.nn.softmax_cross_entropy_with_logits函数。在只有一个正确答案的分类问题中，TensorFlow提供了tf.nn.sparse_softmax_cross_entropy_with_logits函数来进一步加速计算过程。

自定义损失函数：
```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-10 21:55
# @Author  : Storm
# @File    : chapter04-01.py

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# 一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 自定义损失函数
loss_less = 10
loss_more = 1
# tf.where相当于问号表达式，参数一True时选择参数二，否则选择参数三
# tf.greater比较两个张量的大小
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 加一个随机量，加入不可预测的噪音，否则不同损失函数的意义就不大了。噪音均值为0的小量。
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

# 训练神经网络
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        print(sess.run(w1))

```



## 2.神经网络优化

​   TensorFlow提供了灵活的学习率设置方法----指数衰减法，tf.train.exponential_decay函数实现了指数衰减学习率。

​   为了避免过拟合问题，常用的方法是正则化（regularization）。正则化的思想就是在损失函数中加入刻画模型复杂程度的指标。

​   L1正则化和L2正则化：L1正则化会让参数变得稀疏，而L2正则化不会；L1正则化的计算公式不可导，而L2正则化公式可到。

​   通过集合计算一个5层神经网络带L2正则化的损失函数：
```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-11 14:42
# @Author  : Storm
# @File    : chapter04-02.py
# 通过集合计算一个5层神经网络带L2正则化的损失函数

import tensorflow as tf


# 获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名为‘losses’的集合中
def get_weight(shape, lambd):
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # add_to_collection函数将这个新生成变量的L2正则化损失项加入集合
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
    # 返回生成的变量
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
# 定义每一次网络中节点的个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播时最深层的节点，开始就是输入层
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成5层全连接的神经网络
for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用ReLU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    in_dimension = layer_dimension[i]

# 在定义神经网络前向传播的同时已经将所有的L2正则化损失加入了图上的集合
# 这里只需要计算刻画模型在训练数据上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将均方误差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)

# get_collection返回一个列表，这个列表是所有这个集合中的元素
loss = tf.add_n(tf.get_collection('losses'))

```


## 3.滑动平均模型

```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-11 15:16
# @Author  : Storm
# @File    : chapter04-03.py
# 滑动平均模型

import tensorflow as tf

# 定义一个变量用于计算滑动平均
v1 = tf.Variable(0, dtype=tf.float32)
# step变量模拟神经网络中迭代的轮数，可以用于动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类，初始化时给定衰减率0.99和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    # 通过ema.average(v1)获取滑动平均之后变量的值
    print(sess.run([v1, ema.average(v1)]))  # 输出：[0.0, 0.0]

    # 更新变量v1的值到5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值。衰减率为min{0.99，(1+step)/(10+step)=0.1}=0.1
    # v1的滑动平均会被更新为 0.1*0+0.9*5=4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))  # 输出：[5.0, 4.5]

    # 更新step的值为1000
    sess.run(tf.assign(step, 1000))
    # 更新v1的值为10
    sess.run(tf.assign(v1, 10))
    # 更新v1的滑动平均值
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))  # 输出：[10.0, 4.5549998]

    # 再次更新滑动平均值
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))  # 输出：[10.0, 4.6094499]

```


TensorFlow 实战Google深度学习框架（第四章）
