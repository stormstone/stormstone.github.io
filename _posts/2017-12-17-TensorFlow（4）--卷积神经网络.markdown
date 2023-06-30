---
layout:     post
title:      "TensorFlow学习笔记（4）--卷积神经网络"
date:       2017-12-17 21:00:00
author:     "SH"
header-img: "img/post_bg_headset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - TensorFlow
---



## 1.网址

​	Cifar数据集网址：https://www.cs.toronto.edu/~kriz/cifar.html

​	WordNet网址：https://wordnet.princeton.edu/

​	ImageNet网址：http://www.image-net.org/




## 2.简介

​	卷积神经网络和全连接神经网络的唯一区别就在于神经网络中相邻两层的连接方式，卷积神经网络相邻层之间只有部分节点相连。

​	卷积神经网络结构：输入层、卷积层、池化层、全连接层、softmax层。

卷积层：卷积层中每一个节点的输入只是上一层神经网络的一小块，这个小块大小通常为3x3或者5x5。一般来说，通过卷积层处理过的节点矩阵会变得更深。

池化层：池化层神经网络不会改变三维矩阵的深度，但是它可以缩小矩阵的大小。




## 3.卷积层

​	TensorFlow称这个结构为过滤器。过滤器可以将当前层神经网络上的一个子节点矩阵转化为下一层神经网络上的一个单位节点矩阵。单位节点矩阵指的是一个长和宽都为1，单深度不限的节点矩阵。

​	过滤器所处理的节点矩阵的长和宽都是由人工指定的，这个节点矩阵的尺寸也被称之为过滤器的尺寸。常用的过滤器尺寸有3x3或5x5。

​	卷积结构的前向传播过程就是通过将一个过滤器从神经网络当前层的左上角移动到右下角，并且在移动中计算每一个对应的单位矩阵得到的。

​	全0填充（zero-padding），可以使得卷积层前向传播结果矩阵的大小和当前层矩阵保持一致。

​	在卷积神经网络中，每一个卷积层中使用的过滤器中的参数都是一样的。这是卷积神经网络一个非常重要的性质。从直观上理解，共享过滤器的参数可以使得图像上的内容不受位置的影响。共享每一个卷积层中过滤器中的参数可以巨幅减少神经网络上的参数。

​	tf.nn.conv2d 函数来实现卷积层前向传播。
```python
filter_weight = tf.get_variable('weights',[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=0.1))

conv = tf.nn.conv2d(input,filter_weight,strides=[1,1,1,1],padding='SAME')
```
​	input：当前层的节点矩阵，这个矩阵是一个四维矩阵，后面三个维度对应一个节点矩阵，第一维对应一个输入batch。
​	filter_weight：卷积层的权重，前两个维度代表过滤器的尺寸，第三个维度表示当前层的深度，第四个维度表示过滤器的深度。
​	strides：不同维度上的步长，第一维和最后一维的数字要求一定是1，因为卷积层的步长只对矩阵的长和宽有效。
​	padding：填充的方法，SAME表示添加全0填充，VALID表示不添加。




## 4.池化层

​	池化层可以非常有效地缩小矩阵的尺寸，从而减少最后全连接层中的参数。使用池化层既可以加快计算速度也有防止过拟合问题的作用。

​	最大池化层（max pooling），这是被使用最多的池化层结构。还有 平均池化层（average pooling），使用较少。

​	卷积层和池化层中过滤器的移动方式是相似的，唯一的区别在于卷积层使用的过滤器是横跨整个深度的，而池化层使用的过滤器只影响一个深度上的节点。

​	tf.nn.max_pool函数实现了最大池化层的前向传播过程。
```python
pool = tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
```
​	参数和tf.nn.conv2d函数类似，ksize提供了过滤器的尺寸，第一个和最后一个数必须为1，意味着池化层的过滤器是不可以跨不同输入样例或者节点矩阵深度的。在使用中使用得最多的池化层过滤器尺寸为[1,2,3,1]或者[1,3,3,1].
​	strides提供步长信息，第一维和最后一维也只能为1，意味着池化层不能减少节点矩阵的深度或者输入样例的个数。
​	padding提供是否使用全0填充。




## 5.经典卷积网络模型

### LeNet-5模型：
​	LeNet5_infernece.py：
```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-17 18:45
# @Author  : Storm
# @File    : LeNet5_infernece.py

import tensorflow as tf

# 配置神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层的节点数
FC_SIZE = 512


def inference(input_tensor, train, regularizer):
    # 通过使用不同的命名空间来隔离不同层的变量。
    # 第一层卷积层的实现。和标准的LeNet-5模型不大一样，这里定义的卷积层输入为28*28*1的原始MNIST图片像素。
    # 因为卷积层中使用了全0填充，所以输出为28*28*32的矩阵。
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用边长为5,深度为32的过滤器,步长为1,使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层池化层的实现。
    # 选用最大池化层，过滤器边长为2，步长为2，使用全0填充。
    # 这一层输入是上一层的输出，28*28*32，输出为14*14*32。
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 第三层卷积层的实现。输出为14*14*64。
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层池化层的实现。输出为7*7*64。
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 将第四层池化层的输出转化为第五层全连接层的输入格式。
        pool_shape = pool2.get_shape().as_list()
        # 计算将矩阵拉直成向量之后的长度，长宽和深度的乘积。
        # pool_shape[0]为一个batch中数据的个数
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        # 通过tf.reshape函数将第四层的输出变成一个batch的向量。
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 第五次全连接层的实现。输入向量长度为3136，输出为长度为512的向量。
    # dropout随机失活。
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化。
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    # 第六层全连接层的实现。输出为长度为10的向量。通过softmax之后就得到了最后的分类结果。
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit

```
​	LeNet5_train.py:
```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-17 18:46
# @Author  : Storm
# @File    : LeNet5_train.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_infernece
import numpy as np

# 1. 定义神经网络相关的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99


# 2. 定义训练过程
def train(mnist):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        LeNet5_infernece.IMAGE_SIZE,
        LeNet5_infernece.IMAGE_SIZE,
        LeNet5_infernece.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet5_infernece.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = LeNet5_infernece.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 10 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))


# 3. 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()

```

​	经典的用于图片分类问题的卷积神经网络架构：
	输入层 -->（卷积层+ --> 池化层？）+ --> 全连接层+
​	大部分卷积神经网络中一般最多连续使用三层卷积层。池化层虽然可以起到减少参数防止过拟合问题，但是在部分论文中也发现可以直接通过调整卷积层步长来完成。所以有些卷积神经网络中没有池化层。

### Inception-v3模型

​	在LeNet-5模型中，不同卷积层通过串联的方式连接在一起，而Inception-v3模型中的Inception结构是将不同的卷积层通过并联的方式结合在一起。

​	Inception模型同时使用所有不同尺寸的过滤器，然后再将得到的矩阵拼接起来。如果所有的过滤器都使用全0填充且步长为1，那么前向传播得到的结果矩阵的长和宽都与输入矩阵一样，这样经过不同过滤器处理的结果矩阵可以拼接成一个更深的矩阵。

​	Inception-v3模型中使用的Inception模块复杂且多样。总共有46层，由11个Inception模块组成。共有96个卷积层。

​	TensorFlow-Slim工具可以更加简洁的实现一个卷积层。用法：slim.conv2d(input,32,[3,3])。




## 6.迁移学习

​	将一个数据集上训练好的卷积神经网络模块快速转移到另外一个数据集上。​

数据下载地址：
http://download.tensorflow.org/example_images/flower_photos.tgz

谷歌训练好的Inception-v3模型下载地址：
http://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip

```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-17 20:18
# @Author  : Storm
# @File    : chapter06-02.py
# 迁移学习

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 1. 模型和样本路径的设置
# Inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3模型中代表瓶颈层结果的张量名称。在训练模型时，通过tensor.name来获取张量名称
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
# 图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# 下载的训练好的Inception-v3模型目录
MODEL_DIR = 'datasets/inception_dec_2015'
# 下载的训练好的Inception-v3模型名称
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 因为一个训练数据会被使用多次，所以可以将原始图像通过Inception-v3模型计算得到的特征向量保存在文件中
# 下面的变量定义了这些文件的存放的地址
CACHE_DIR = 'datasets/bottleneck'
# 图片数据
INPUT_DATA = 'datasets/flower_photos'

# 验证数据的百分比
VALIDATION_PERCENTAGE = 10
# 测试数据的百分比
TEST_PERCENTAGE = 10

# 2. 神经网络参数的设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100


# 3.把样本中所有的图片列表并按训练、验证、测试数据分开
# testing_percentage和validation_percentage参数指定测试数据和验证数据的大小。
def create_image_lists(testing_percentage, validation_percentage):
    # 得到的所有图片都存在result这个字典里。key为类别的名称，value也是一个字典，存储了所有图片名称。
    result = {}
    # 获取当前目录下所有的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # 得到的第一个目录是当前目录，不需要考虑
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        # 通过目录名获取类别的名称
        label_name = dir_name.lower()

        # 初始化当前类别的训练数据集、测试数据集和验证数据集
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)

            # 随机划分数据为训练、测试、验证数据集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据放入结果字典。
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


# 4.定义函数通过类别名称、所属数据集和图片编号获取一张图片的地址。
def get_image_path(image_lists, image_dir, label_name, index, category):
    # 获取给定类别中所有图片的信息。
    label_lists = image_lists[label_name]
    # 根据所属数据集的名称获取集合中全部图片的信息。
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图片的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # 最终的地址为数据根目录的地址加上类别的文件夹加上图片的名称。
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


# 5.定义函数获取Inception - v3模型处理之后的特征向量的文件地址。
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'


# 6.定义函数使用加载的训练好的Inception - v3模型处理一张图片，得到这个图片的特征向量。
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # 这个过程实际上就是将当前图片作为输入计算瓶颈张量的值。
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量。
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 7.定义函数会先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件。
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件的路径。
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    # 如果这个特征向量文件不存在，则通过Inception-v3模型来计算特征向量，并存入文件。
    if not os.path.exists(bottleneck_path):
        # 获取原始的图片路径。
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        # 获取图片内容。
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # 通过Inception-v3模型计算特征向量。
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        # 将计算得到的特征向量存入文件。
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 直接从文件中获取图片相应的特征向量。
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


# 8.这个函数随机获取一个batch的图片作为训练数据。
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        # 随机一个类别和图片的编号加入当前的训练数据。
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


# 9.这个函数获取全部的测试数据，并计算正确率。
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # 枚举所有的类别和每个类别中的测试图片。
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            # 通过Inception-v3模型计算图片对应的特征向量，并将其加入最终数据的列表。
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor,
                                                  bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


# 10.定义主函数。
def main():
    # 获取所有图片。
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())

    # 读取已经训练好的Inception-v3模型。
    # Google将训练好的模型保存在GraphDef Protocol Buffer中，里面保存了每一个节点取值的计算方法以及变量的取值。
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 加载读取的Inception-v3模型，并返回数据输入对应的张量以及计算瓶颈层结果所对应的张量。
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义新的神经网络输入
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    # 定义新的标准答案输入。
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    # 定义一层全链接层
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 定义交叉熵损失函数。
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # 训练过程。
        for i in range(STEPS):

            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
                      (i, BATCH, validation_accuracy * 100))

        # 在最后的测试数据上测试正确率。
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    main()

```


TensorFlow 实战Google深度学习框架（第六章）