# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/7/3 20:05
from pylab import *
import numpy as np

mpl.rcParams['font.sans-serif'] = ['SimHei']    # 支持中文输出

#折线图
x = ["1", "2", "3", "4", "5", "Average"]
# x = np.arange(1000, 31000, 2000)
nb_test_accuracy = [0.87038081, 0.8733812, 0.87493148, 0.87503155, 0.87318127, 0.873381262]
cnn_test_accuracy = [0.9374843711076781, 0.9347336831972819, 0.9497374343734946, 0.9424856214202563, 0.9502375591662742, 0.942935733852997]
plt.plot(x, nb_test_accuracy, 's-', color='r', label="nb test accuracy")#s-:方形
plt.plot(x, cnn_test_accuracy, 'o-', color='g', label="cnn test accuracy")#o-:圆形
plt.xticks(x)
plt.xlabel("Iteration")#横坐标名字
plt.ylabel("Value")#纵坐标名字
plt.legend(loc="best")#图例
plt.show()