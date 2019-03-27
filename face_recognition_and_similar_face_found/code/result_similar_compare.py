# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/7/5 16:39
import numpy as np
import matplotlib.pyplot as plt
precision = [28.59, 41.00, 28.90, 32.00]
x = np.arange(4)
total_width, n = 0.8, 4    # 有多少个类型，只需更改n即可
width = total_width / n
x = x - (total_width - width) / 2
plt.bar(x, precision, color="g", width=width, label='accuracy')
plt.xlabel("algorithm")
plt.ylabel("value(%)")
plt.legend(loc="best")

for xx, yy in zip(x,precision):
    plt.text(xx, yy+0.1, str(yy), ha='center')

plt.xticks([-0.3, 0.7, 1.7, 2.7], ['NB', 'RF', 'CNN', 'CNN with augment'])
my_y_ticks = np.arange(25, 45, 2)
plt.ylim((25, 45))
plt.yticks(my_y_ticks)
plt.show()
