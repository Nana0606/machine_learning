# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/6/16 10:45
from pylab import *
import numpy as np

mpl.rcParams['font.sans-serif'] = ['SimHei']    # 支持中文输出

#折线图
x = ["1", "2", "3", "4", "5", "Average"]
# x = np.arange(1000, 31000, 2000)
train_score = [0.0940468285462382, 0.09305162906296507, 0.08970361036693383, 0.08524982205026596, 0.08821227992120452, 0.09005283398952153]
train_accuracy = [0.9644955618930826, 0.9656832103491396, 0.9621827728540573, 0.9662457806704298, 0.9656207025356629, 0.9648456056604744]
test_score = [0.1561250634851024, 0.16276330777326803, 0.13061561616369116, 0.1612305216180083, 0.12634601775080778, 0.14741610535817554]
test_accuracy = [0.9374843711076781, 0.9347336831972819, 0.9497374343734946, 0.9424856214202563, 0.9502375591662742, 0.942935733852997]
plt.plot(x, train_score, 's-', color='r', label="train score")#s-:方形
plt.plot(x, train_accuracy, 'o-', color='r', label="train accuracy")#o-:圆形
plt.plot(x, test_score, 's-', color='g', label="test score")#s-:方形
plt.plot(x, test_accuracy, 'o-', color='g', label="test accuracy")#o-:圆形
plt.xticks(x)
plt.xlabel("Iteration")#横坐标名字
plt.ylabel("Value")#纵坐标名字
plt.legend(loc="best")#图例
plt.show()
