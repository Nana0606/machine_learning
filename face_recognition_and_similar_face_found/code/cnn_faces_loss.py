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
loss= [15.3157, 15.2681, 15.1743, 6.5931, 0.5722, 0.1932, 0.1340, 0.1114, 0.0219, 0.0727]
plt.plot(x, loss, 's-', color='r', label="loss")#s-:方形
plt.xticks(x)
plt.xlabel("Epoch")#横坐标名字
plt.ylabel("Value")#纵坐标名字
plt.legend(loc="best")#图例
plt.show()
