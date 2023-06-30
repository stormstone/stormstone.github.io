---
layout:     post
title:      "TensorFlow学习笔记（1）--基本概念"
date:       2017-12-10 21:00:00
author:     "SH"
header-img: "img/post_bg_headset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - TensorFlow
---



## 1.TensorFlow计算模型--计算图

​    TensorFlow中的所有计算都会被转化为计算图上的节点。

​    TensorFlow----Tensor和Flow。Tensor就是张量，在TensorFlow中，张量可以简单的理解为多为数组。如果说TensorFlow的Tensor表明了它的数据结构，那么flow则体现了它的计算模型。Flow中文翻译就是“流”，它直观表达了张量之间通过计算相互转化的过程。

​    TensorFlow是一个通过计算图的形式来表达计算的编程系统。TensorFlow中的每一个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系。

​    不同计算图上的张量和运算都不会共享。
```python
g1 = tf.Graph()
with g1.as_default():
    # 在计算图g1中定义变量“v”,并设置初始值为0。
    v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer())

g2 = tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量“v”,并设置初始值为1。
    v = tf.get_variable("v", shape=[1], initializer=tf.ones_initializer())

# 在计算图g1中读取变量“v”的值。
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        # 在计算图g1中，取变量“v”的值为0，输出为[0.]。
        print(sess.run(tf.get_variable("v")))

# 在计算图g2中读取变量“v”的值。
with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        # 在计算图g2中，取变量“v”的值为1，输出为[1.]。
        print(sess.run(tf.get_variable("v")))

```

​    TensorFlow中的计算图不仅可以用来隔离张量和计算，它还提供了管理张量和计算的机制。计算图可以通过tf.Graph.device函数来指定运行计算的设备。




## 2.TensorFlow数据模型--张量

​    TensorFlow中所有的数据都通过张量的形式来表示。

​    张量在TensorFlow中的实现并不是直接采用数组的形式，它只是对TensorFlow中运算结果的引用。在张量中并没有保存真正的数字，保存的是如何得到这些数字的计算过程。

```python
# tf.constant是一个计算，这个计算的结果为一个张量，保存在变量a中。
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = tf.add(a, b,name="add")
print(result)
'''
输出为：
    Tensor("add:0", shape=(2,), dtype=float32)
'''
'''
一个张量中主要保存了三个属性：名字（name）、维度（shape）、类型（type）。
'''
```

​    第一个属性name也给出了这个张量是如何计算出来的。“node:src_output” 的形式，node为节点的名称，src_output表示当前张量来自节点的第几个输出。

​    每一个张量会有一个唯一的类型，TensorFlow会对参与运算的所有张量进行类型检查，类型不匹配会报错。所以一般建议通指定dtype来明确指出变量或者常量的类型。
TensorFlow支持14种类型：
```python
#实数（tf.float32,tf.float64）、
#整数（tf.int8,tf.int16,tf.int64,tf.uint8）、
#布尔型（tf.bool）、
#复数（tf.complex64,tf.complex128）。
```



## 3.TensorFlow运行模型--会话

​    会话拥有并管理TensorFlow程序运行时的所有资源。当所有计算完成之后需要关闭会话来帮助系统回收资源，否则就可能出现资源泄露问题。
```python
# 创建一个会话
sess = tf.Session()
# 使用这个创建好的会话来得到结果
sess.run(result)
# 关闭会话使得本次运行中的资源可以被释放
sess.close()
```
​    可以通过python的上下文管理器来使用会话。
```python
# 创建一个会话，并通过python中的上下文管理器来管理这个会话
with tf.Session() as sess:
    sess.run(result)
# 不在需要调用Session.close()函数来关闭会话，
# 当上下文退出时会话关闭和资源释放自动完成
```

​    TensorFlow不会自动生成默认的会话，需要手动指定。当默认会话被指定后可以通过tf.Tensor.eval函数来计算一个张量的值。
```python
sess = tf.Session()
with sess.as_default():
    print(result.eval())
# 以下代码可以完成相同的功能
print(sess.run(result))
print(result.eval(session=sess))
```
​    使用tf.InteractiveSession函数可以自动将生成的会话注册为默认会话。
```python
sess = tf.InteractiveSession()
print(result.eval())
sess.close()
```
​    通过ConfigProto Protocol Buffer 来配置。
```python
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
```



## 4.TensorFlow训练神经网络模型
```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-10 20:06
# @Author  : Storm
# @File    : chapter03-02.py

import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络的前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128

X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行TensorFlow程序
with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    '''
    w1 = [[-0.81131822  1.48459876  0.06532937]
     [-2.4427042   0.0992484   0.59122431]]
    w2 = 
    [[-0.81131822]
     [ 1.48459876]
     [ 0.06532937]]
    '''

    # 设定训练轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        # 每1000次训练计算所有数据上的交叉熵
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy))
        '''
        After 0 training step(s),cross entropy on all data is 0.0674925
        After 1000 training step(s),cross entropy on all data is 0.0163385
        After 2000 training step(s),cross entropy on all data is 0.00907547
        After 3000 training step(s),cross entropy on all data is 0.00714436
        After 4000 training step(s),cross entropy on all data is 0.00578471
        
        交叉熵逐渐变小。
        '''

    print(sess.run(w1))
    print(sess.run(w2))
    '''
    w1=[[-1.96182752  2.58235407  1.68203771]
     [-3.46817183  1.06982315  2.11788988]]
    w2=[[-1.82471502]
     [ 2.68546653]
     [ 1.41819501]]
    '''

```


TensorFlow 实战Google深度学习框架（第三章）
