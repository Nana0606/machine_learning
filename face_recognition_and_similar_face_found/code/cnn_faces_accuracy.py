# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/7/5 15:43
from pylab import *
import numpy as np

mpl.rcParams['font.sans-serif'] = ['SimHei']    # 支持中文输出

#折线图
x = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
# x = np.arange(1000, 31000, 2000)
accuracy = [41.0, 52.7, 54.7, 20.31, 84.96, 94.53, 96.88, 97.85, 99.41, 98.24]
plt.plot(x, accuracy, 'o-', color='g', label="accuracy")#o-:圆形
plt.xticks(x)
plt.xlabel("Epoch")#横坐标名字
plt.ylabel("Value(%)")#纵坐标名字
plt.legend(loc="best")#图例
plt.show()
