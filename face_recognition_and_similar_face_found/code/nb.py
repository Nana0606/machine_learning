# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/7/5 9:36
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

"""
使用NB进行相似人脸发现
"""

BASE_DIR= "../data/faces.tar/faces"
LENGTH = 120
WIDTH = 128
NUM_CLASSES = 4

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
            for fname in sorted(os.listdir(path)):
                if not fname.endswith("_2.pgm") and not fname.endswith("_4.pgm"):
                    emoij = {
                        "angry": 0,
                        "happy": 1,
                        "neutral": 2,
                        "sad": 3
                    }
                    fpath = os.path.join(path, fname)
                    data = readpgm(fpath)    # data[0]为数据，data[1]为shape
                    # print(data[1])
                    if data[1]==(LENGTH, WIDTH):    # 图片(120, 128)
                        # X.append(data[0].reshape(LENGTH, WIDTH))
                        X.append(data[0])
                        str = fname.split("_")[2]
                        y.append(emoij[str])
    return X, y


if __name__ == '__main__':
    X, y = read_data()
    X_data = np.array(X)
    y_data = np.array(y)
    print("X的格式为： ", X_data.shape)
    print("y的格式为： ", y_data.shape)

    # 使用MultinomialNB进行相似人脸发现
    param_range = np.logspace(-6, -1, 5)
    train_accuracy, validation_accuracy = validation_curve(MultinomialNB(), X_data, y_data, cv=5, param_name="alpha",
                                                           param_range=param_range, scoring="accuracy")
    print("train_score is: *************************************")
    print(train_accuracy)
    print("validation_score is: *************************************")
    print(validation_accuracy)

    train_accuracy_mean = train_accuracy.mean(1)
    train_accuracy_std = train_accuracy.std(1)
    validation_accuracy_mean = validation_accuracy.mean(1)
    validation_accuracy_std = validation_accuracy.std(1)

    print("validation_accuracy average: ", validation_accuracy_mean)

    print("train_accuracy_mean: ", train_accuracy_mean.mean(0))
    print("validation_accuracy_mean: ", validation_accuracy_mean.mean(0))
