---
layout:     post
title:      "TensorFlow学习笔记（5）--图像数据处理"
date:       2017-12-20 16:00:00
author:     "SH"
header-img: "img/post_bg_headset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - TensorFlow
---



## 1.TFRecord数据格式

​	使用TFRecord将MNIST输入数据转化为TFRecord的格式：
```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-18 11:56
# @Author  : Storm
# @File    : chapter07-01.py
# 使用TFRecord将MNIST输入数据转化为TFRecord的格式.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# 1. 将输入转化成TFRecord格式并保存。
# 定义函数转化变量类型。
# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 读取mnist数据。
mnist = input_data.read_data_sets("datasets/MNIST_data", dtype=tf.uint8, one_hot=True)
images = mnist.train.images
# 训练数据所对应的正确答案，可以作为一个属性保存在TFRecord中。
labels = mnist.train.labels
# 训练数据的图像分辨率，可以作为Example中的一个属性。
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出TFRecord文件的地址。
filename = "datasets/Records/output.tfrecords"
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    # 将图像矩阵转化成一个字符串。
    image_raw = images[index].tostring()

    # 将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构。
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())
writer.close()
print("TFRecord文件已保存。")


# 2. 读取TFRecord文件
# 读取文件。
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表。
filename_queue = tf.train.string_input_producer(["datasets/Records/output.tfrecords"])
# 从文件中读出一个样例，也可以使用read_up_to函数一次性读取多个样例。
_, serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

# tf.decode_raw可以将字符串解析成图像对应的像素数组。
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
```



## 2.图像数据处理

### 1）图像编码处理
```python
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# 路径
IMG_DIR = 'datasets/cat'
IMG_NAME = 'cat.jpg'

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile(os.path.join(IMG_DIR, IMG_NAME), 'rb').read()

with tf.Session() as sess:
    # 将图像使用jpeg的格式解码从而得到对应的三维矩阵。
    img_data = tf.image.decode_jpeg(image_raw_data)

    print(img_data.eval())
    # 显示图像
    plt.imshow(img_data.eval())
    plt.show()

    # 将数据的类型转化成实数方便处理
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)

    # 将表示一张图像的三维矩阵从新安装jpeg格式编码存入文件中。
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile(os.path.join(IMG_DIR, 'cat_jpeg.jpeg'), 'wb')as f:
        f.write(encoded_image.eval())
```

### 2）图像大小调整
```python
# 2.图像大小调整
    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
    print("Digital type: ", resized.dtype)
    cat = np.asarray(resized.eval(), dtype='uint8')
    # tf.image.convert_image_dtype(rgb_image, tf.float32)
    plt.imshow(cat)
    plt.savefig("datasets/cat/cat-大小调整.jpg")
    plt.show()

    # 裁剪和填充图片
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
    plt.imshow(croped.eval())
    plt.savefig("datasets/cat/cat-裁剪图片.jpg")
    plt.show()
    plt.imshow(padded.eval())
    plt.savefig("datasets/cat/cat-填充图片.jpg")
    plt.show()
    # 截取图片
    central_cropped = tf.image.central_crop(img_data, 0.5)
    plt.imshow(central_cropped.eval())
    plt.savefig("datasets/cat/cat-截取图片.jpg")
    plt.show()

```
method参数取值与对应的调整算法：
  0 ：双线性插值法<br>
  1 ：最近邻居法<br>
  2 ：双三次插值法<br>
  3 ：面积插值法<br>

### 3）图像翻转
```python
    # 3.图像翻转
    # 上下翻转
    flipped1 = tf.image.flip_up_down(img_data)
    plt.imshow(flipped1.eval())
    plt.savefig("datasets/cat/cat-上下翻转.jpg")
    # 左右翻转
    flipped2 = tf.image.flip_left_right(img_data)
    plt.imshow(flipped2.eval())
    plt.savefig("datasets/cat/cat-左右翻转.jpg")

    # 对角线翻转
    transposed = tf.image.transpose_image(img_data)
    plt.imshow(transposed.eval())
    plt.savefig("datasets/cat/cat-对角翻转.jpg")
    plt.show()

    # 以一定概率上下翻转图片。
    flipped3 = tf.image.random_flip_up_down(img_data)
    plt.imshow(flipped3.eval())
    plt.savefig("datasets/cat/cat-一定概率上下翻转.jpg")
    # 以一定概率左右翻转图片。
    flipped4 = tf.image.random_flip_left_right(img_data)
    plt.imshow(flipped4.eval())
    plt.savefig("datasets/cat/cat-一定概率左右翻转.jpg")

```

### 4）图像色彩调整
```python
# 4.图像色彩调整
    # 将图片的亮度-0.5。
    adjusted = tf.image.adjust_brightness(img_data, -0.5)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-亮度减0.5.jpg")

    # 将图片的亮度+0.5
    adjusted = tf.image.adjust_brightness(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-亮度加0.5.jpg")

    # 在[-max_delta, max_delta)的范围随机调整图片的亮度。
    adjusted = tf.image.random_brightness(img_data, max_delta=0.5)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-随机调整亮度.jpg")

    # 将图片的对比度-5
    adjusted = tf.image.adjust_contrast(img_data, -5)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-对比度减0.5.jpg")

    # 将图片的对比度+5
    adjusted = tf.image.adjust_contrast(img_data, 5)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-对比度加0.5.jpg")

    # 在[lower, upper]的范围随机调整图的对比度。
    lower = 1
    upper = 5
    adjusted = tf.image.random_contrast(img_data, lower, upper)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-随机调整对比度.jpg")
    plt.show()

    # 添加色相和饱和度
    adjusted = tf.image.adjust_hue(img_data, 0.1)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-色相加0.1.jpg")
    adjusted = tf.image.adjust_hue(img_data, 0.3)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-色相加0.3.jpg")
    adjusted = tf.image.adjust_hue(img_data, 0.6)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-色相加0.6.jpg")
    adjusted = tf.image.adjust_hue(img_data, 0.9)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-色相加0.9.jpg")

    # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间。
    max_delta = 0.3
    adjusted = tf.image.random_hue(img_data, max_delta)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-随机调整色相.jpg")

    # 将图片的饱和度-5。
    adjusted = tf.image.adjust_saturation(img_data, -5)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-饱和度减5.jpg")
    # 将图片的饱和度+5。
    adjusted = tf.image.adjust_saturation(img_data, 5)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-饱和度加5.jpg")
    # 在[lower, upper]的范围随机调整图的饱和度。
    adjusted = tf.image.random_saturation(img_data, lower, upper)
    plt.imshow(adjusted.eval())
    plt.savefig("datasets/cat/cat-随机调整饱和度.jpg")
    plt.show()

```

### 5）处理标注框
```python
# 5.处理标注框
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes)

    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)

    distorted_image = tf.slice(img_data, begin, size)
    plt.imshow(distorted_image.eval())
    plt.savefig("datasets/cat/cat-标注框.jpg")
    plt.show()
```

### 6）图像预处理完整样例
​	从图像片段截取，到图像大小调整再到图像翻转以及色彩调整的整个过程。

```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-19 15:25
# @Author  : Storm
# @File    : chapter07-03.py
# 图像预处理完整样例

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 1.随机调整图片的色彩，定义两种顺序。
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)


# 2.对图片进行预处理，将图片转化成神经网络的输入层数据。
def preprocess_for_train(image, height, width, bbox):
    # 查看是否存在标注框。
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机的截取图片中一个块。
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图片调整为神经网络输入层的大小。
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image


# 3.读取图片。
image_raw_data = tf.gfile.FastGFile("datasets/cat/cat.jpg", "rb").read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    for i in range(9):
        result = preprocess_for_train(img_data, 299, 299, boxes)
        plt.imshow(result.eval())
        plt.savefig("datasets/cat/cat-完整样例-" + str(i))
        plt.show()

```



## 3.多线程输入数据处理框架

​	在TensorFlow中，队列不仅是一种数据结构，更提供了多线程机制。

### 1）队列与多线程
​	队列和变量类似，都是计算图上有状态的节点。
```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-19 15:33
# @Author  : Storm
# @File    : chapter07-04.py
# 队列操作

import tensorflow as tf
import numpy as np
import threading
import time

# 1. 创建队列，并操作里面的元素。
# 指定队列中最多可以保存两个元素，并指定类型为整数。
q = tf.FIFOQueue(2, "int32")
# 初始化队列中的元素。
init = q.enqueue_many(([0, 10],))
# 使用dequeue函数将队列中的第一个元素取出。
x = q.dequeue()
y = x + 1
# 将加一后的值再重新加入队列。
q_inc = q.enqueue([y])
with tf.Session() as sess:
    # 运行队列初始化操作。
    init.run()
    for _ in range(5):
        # 运行q_inc将执行数据出队列，出队的元素加一，重新加入队列的过程。
        v, _ = sess.run([x, q_inc])
        print(v)


# 2. 每隔1秒判断是否需要停止并打印自己的ID。
def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stoping from id: %d\n" % worker_id, )
            coord.request_stop()
        else:
            print("Working on id: %d\n" % worker_id, )
        time.sleep(1)


# 3. 创建、启动并退出线程。
coord = tf.train.Coordinator()
# 声明创建5个线程
threads = [threading.Thread(target=MyLoop, args=(coord, i,)) for i in range(5)]
for t in threads: t.start()
coord.join(threads)

# 4.多线程队列操作
queue = tf.FIFOQueue(100, "float")
# 定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])
# tf.train.QueueRunner来创建多个线程运行队列的入队操作。
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)
# 将定义过的QueueRunner加入TensorFlow计算图上指定的集合。
tf.train.add_queue_runner(qr)
# 定义出队操作
out_tensor = queue.dequeue()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    # 使用tf.train.QueueRunner时，需要调用tf.train.start_queue_runners来启动所有线程。
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for _ in range(3): print(sess.run(out_tensor)[0])
    coord.request_stop()
    coord.join(threads)

```

### 2）输入文件队列
```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-19 15:40
# @Author  : Storm
# @File    : chapter07-05.py
# 输入文件队列

import tensorflow as tf


# 1.生成文件存储样例数据。
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


num_shards = 2
instances_per_shard = 2
for i in range(num_shards):
    filename = ('datasets/Records/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
    # 将Example结构写入TFRecord文件。
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(instances_per_shard):
        # Example结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本。
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()

# 2. 读取文件。
files = tf.train.match_filenames_once("datasets/Records/data.tfrecords-*")
filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64),
    })
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    print(sess.run(files))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(6):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)
```

### 3）组合训练数据
```python
# 3. 组合训练数据（Batching）
example, label = features['i'], features['j']
batch_size = 2
capacity = 1000 + 3 * batch_size
example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)
    coord.request_stop()
    coord.join(threads)


```

### 4）输入数据处理框架
```python
# -*- coding: utf-8 -*-
# @Time    : 2017-12-19 15:43
# @Author  : Storm
# @File    : chapter07-06.py
# 输入数据处理框架

import tensorflow as tf

# 1.创建文件列表，通过文件列表创建输入文件队列。
files = tf.train.match_filenames_once("datasets/Records/output.tfrecords")
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 3.解析TFRecord文件里的数据。
# 读取文件。
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
retyped_images = tf.cast(decoded_images, tf.float32)
labels = tf.cast(features['label'], tf.int32)
# pixels = tf.cast(features['pixels'],tf.int32)
images = tf.reshape(retyped_images, [784])


# 4.将文件以100个为一组打包。
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size

image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                  batch_size=batch_size,
                                                  capacity=capacity,
                                                  min_after_dequeue=min_after_dequeue)


# 5.训练模型。
def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2


# 模型相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 5000

weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

y = inference(image_batch, weights1, biases1, weights2, biases2)

# 计算交叉熵及其平均值
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label_batch)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

# 损失函数的计算
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
regularaztion = regularizer(weights1) + regularizer(weights2)
loss = cross_entropy_mean + regularaztion

# 优化损失函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 初始化会话，并开始训练过程。
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 循环的训练神经网络。
    for i in range(TRAINING_STEPS):
        if i % 1000 == 0:
            print("After %d training step(s), loss is %g " % (i, sess.run(loss)))
        sess.run(train_step)
    coord.request_stop()
    coord.join(threads)

```


TensorFlow 实战Google深度学习框架（第七章）