# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/6/17 17:45
from PIL import Image
import os
import numpy as np


def read_img():
    """
    读取模式为P5的.pgm图片
    :return: 
    """
    im = Image.open("../data/faces.tar/faces/an2i/an2i_left_angry_open_4.pgm")
    im.show()
    print(im.size)


def read_image():
    """
    读取图片存储到数组中
    :return: 
    """
    dirs = os.listdir("./pic")
    Y = [] #label
    X = [] #data
    print(len(dirs))
    for filename in dirs:
        label = int(filename.split('_')[0])
        Y.append(label)
        im = Image.open("./pic//{}".format(filename)).convert('1')
        mat = np.asarray(im) #image 转矩阵
        X.append(mat)
    return np.array(X),np.array(Y)


if __name__ == "__main__":
    read_img()