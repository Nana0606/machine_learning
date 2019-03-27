# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/5/19 9:44
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

print("Training data size: ", mnist.train.num_examples)
print("Validation data size: ", mnist.validation.num_examples)
print("Testing data size：", mnist.test.num_examples)
print("Handled training data: ", mnist.train.images.shape)
# print("Example training data: ", mnist.train.images[0])
print("Example training data label: ", mnist.train.labels[0])
print("mnist.validation.images: ", mnist.validation.images.shape)
print("mnist.validation.labels: ", mnist.validation.labels.shape)
print("mnist.test.images: ", mnist.test.images.shape)
print("mnist.test.labels: ", mnist.test.labels.shape)
