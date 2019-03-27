# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/5/22 22:09
#柱状图
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']    # 支持中文输出

x = ["使用所有优化", "不用滑动平均", "不用隐藏层", "不用指数衰减学习率", "不用正则化", "不用激活函数", "使用Adam"]
# data = [0.9842, 0.9849, 0.927, 0.9754, 0.98,  0.9266, 0.9845]
data = [0.9842, 0.9828, 0.927, 0.9754, 0.98, 0.9266, 0.9845]

plt.bar(x, data, width=0.4, color='#FF8C00')    # #FF8C00
plt.xticks([0, 1, 2, 3, 4, 5, 6], x)
plt.xlabel("不同模型精度")
plt.ylabel("精度")
for i in range(len(x)):
    plt.text(x[i], data[i], data[i], ha='center', va='bottom')
# plt.xticks(x)
my_y_ticks = np.arange(0.9, 1, 0.01)     # 设置纵坐标间隔
plt.ylim((0.9, 1))    # 设置纵坐标范围
plt.yticks(my_y_ticks)
plt.show()

