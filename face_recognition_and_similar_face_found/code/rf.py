# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/7/5 14:30
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

"""
使用随机森林进行相似人脸发现
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


def readData():
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
    X, y = readData()
    X_data = np.array(X)
    y_data = np.array(y)
    print("X的格式为： ", X_data.shape)
    print("y的格式为： ", y_data.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)
    clf = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=33)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_pred, y_test, target_names=["angry", "happy", "neutral", "sad"]))

