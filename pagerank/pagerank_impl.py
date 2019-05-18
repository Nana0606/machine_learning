# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/7/27 11:04
"""
实现pagerank库计算
"""
import numpy as np

# 图的邻接矩阵
a = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
], dtype=float)

def transition_matrix(a):
    """
    根据邻接矩阵求转移矩阵 
    :param a: 邻接矩阵
    :return: 转移矩阵
    """
    p = np.zeros(a.shape, dtype=float)
    for i in range(0, a.shape[0]):
        for j in range(0, a.shape[1]):
            p[i][j] = a[i][j] / a[i].sum()   # 对于每一个元素，其转移概率为当前元素除以所在行元素之和
    return p

def initializeV(N):
    """
    初始化v矩阵，即重要性矩阵
    :param N: 图中节点数
    :return: 初始化之后的v矩阵
    """
    v = np.zeros((N, 1), dtype=float)
    for i in range(0, N):
        v[i] = 1 / N    # 每一个元素值相等，大小为1/N
    return v

def pagerank(p, v):
    """
    pagerank代码，使用迭代的思想，直到达到平稳状态，使用公式v_{n+1} = p^T * v_{n}
    :param p: 转移矩阵
    :param v: 初始化的v矩阵
    :return: 平稳状态v矩阵
    """
    v_current = v
    v_then = np.dot(np.transpose(p), v)     # v1 = p^T*v0
    while(np.abs(v_then-v_current).all() >= 0.000001):   # 对应位置元素之差大于0.000001时，说明状态未收敛，继续运行
        v_current = v_then
        v_then = np.dot(np.transpose(p), v_then)   # v2 = p^T*v1
        print("v_then is::", v_then)
    return v_then

if __name__ == '__main__':
    p = transition_matrix(a)
    v = initializeV(a.shape[0])
    v_then = pagerank(p, v)
    print("转移矩阵是：", p)
    print("初始化v向量是：", v)
    print("最后的平稳状态是", v_then)