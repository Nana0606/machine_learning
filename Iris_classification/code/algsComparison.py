# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/4/10 15:19
'''
function：
    draw a bar graph to compare the performance of those three algorithms more clearly.
'''
#柱状图
import numpy as np
import matplotlib.pyplot as plt
precision = [1.00, 0.92, 1.00]
recall = [1.00, 0.89, 1.00]
f1_score = [1.00, 0.89, 1.00]
x = np.arange(3)
total_width, n = 0.8, 3    # 有多少个类型，只需更改n即可
width = total_width / n
x = x - (total_width - width) / 2
plt.bar(x, precision, color="r", width=width, label='precision')
plt.bar(x + width, recall, color="y", width=width, label='recall')
plt.bar(x + 2 * width, f1_score, color="c", width=width, label='f1_score')
plt.xlabel("algorithm")
plt.ylabel("value")
plt.legend(loc="best")
plt.xticks([0, 1, 2], ['LR', 'DT', 'MLP'])
my_y_ticks = np.arange(0.8, 1.02, 0.02)
plt.ylim((0.8, 1.02))
plt.yticks(my_y_ticks)
plt.show()
