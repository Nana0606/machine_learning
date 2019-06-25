# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/6/17 19:27
"""
使用CNN进行人脸识别
"""
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import np_utils
import keras

BASE_DIR= "../data/faces.tar/faces"
LENGTH = 120
WIDTH = 128
NUM_CLASSES = 20

def readpgm(name):
    """
    PIL只能处理.pgm数据的P4、P5和P6，本函数将.pgm文件的P2格式读取出来
    :param name: 文件名
    :return: 图片的数据表示，图片的维度
    """
    with open(name) as f:
        lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    # Makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2'

    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])

    return (np.array(data[3:]),(data[1],data[0]),data[2])

def read_data():
    """
    将faces文件中的数据读取出来，并获取标签信息
    :return: X，一个3维矩阵（图片数，LENGTH,WIDTH），y标签，是一个长度为图片数的向量
    """
    X = []
    labels_index = {}     # dictionary mapping label name to numeric id
    y = []    # list of label ids
    for dir_name in sorted(os.listdir(BASE_DIR)):
        path = os.path.join(BASE_DIR, dir_name)
        if os.path.isdir(path):
            label_id = len(labels_index)   # 文件夹id
            labels_index[dir_name] = label_id
            for fname in sorted(os.listdir(path)):
                if not fname.endswith("_2.pgm") and not fname.endswith("_4.pgm"):
                    fpath = os.path.join(path, fname)
                    data = readpgm(fpath)    # data[0]为数据，data[1]为shape
                    # print(data[1])
                    if data[1]==(LENGTH, WIDTH):    # 图片(120, 128)
                        X.append(data[0].reshape(LENGTH, WIDTH))
                        y.append(label_id)
    return X, y

def cnn_model():
    """
    构建CNN网络，共设置了4层卷积层和2层全连接层
    :return: 
    """
    model = Sequential()
    model.add(Convolution2D(
        input_shape = (LENGTH, WIDTH, 1),   # (?, 120, 128, 1)
        filters=32,
        kernel_size=3,     # 卷积窗口的大小
        strides=1,      # Convolution2D中的strides设置为1，步长，每次走的跳步
        padding="same",
        data_format="channels_last"
    ))    # (?, 120, 128, 32)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,    # MaxPooling2D中的strides设置成2
        data_format="channels_last"
    ))     # (?, 60, 64, 32)
    model.add(Convolution2D(32, 3, strides=1, padding='same', data_format='channels_last'))   # (?, 60, 64, 32)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2, data_format='channels_last'))    # (?, 30, 32, 32)
    model.add(Convolution2D(32, 3, strides=1, padding='same', data_format='channels_last'))  # (?, 30, 32, 32)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2, data_format='channels_last'))  # (?, 15, 16, 32)
    model.add(Convolution2D(32, 3, strides=1, padding='same', data_format='channels_last'))  # (?, 15, 16, 32)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2, data_format='channels_last'))  # (?, 8, 8, 32)
    # model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(8*8*32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer=keras.optimizers.RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model



if __name__ == '__main__':
    X, y = read_data()
    X_data = np.array(X)
    y_data = np.array(y)
    print("X的格式为： ", X_data.shape)
    print("y的格式为： ", y_data.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)

    model = cnn_model()
    print('\nTraining ------------')  # 从文件中提取参数，训练后存在新的文件中

    print(X_train.shape)
    X_train = np.reshape(X_train, (-1, X_train.shape[1], X_train.shape[2], 1))   # 需要4维数据，这里将3维转化为4维，第四个元素1表示是灰度图
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES)   # 将标签信息one-hot
    np.random.seed(33)  # 设置随机种子
    model.fit(X_train, y_train, epochs=10, batch_size=16)  # 正式训练数据

    print("训练完成********************8")
    X_test = np.reshape(X_test, (-1, X_test.shape[1], X_test.shape[2], 1))
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES)
    loss, accuracy = model.evaluate(X_test, y_test)

    model.save_weights('cnn_model' + '.h5')

    print("loss is: ", loss)
    print("accuracy is: ", accuracy)







