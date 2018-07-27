# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/7/27 12:33
"""
使用networkX库实现pagerank库计算
"""
import networkx as nx
import matplotlib.pyplot as plt

def buildDiGraph(edges):
    """
    初始化图
    :param edges: 存储有向边的列表
    :return: 使用有向边构造完毕的有向图
    """
    G = nx.DiGraph()   # DiGraph()表示有向图
    for edge in edges:
        G.add_edge(edge[0], edge[1])   # 加入边
    return G

if __name__ == '__main__':
    edges = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"), ("A", "D"), ("D", "C")]
    G = buildDiGraph(edges)

    # 将图形画出
    layout = nx.spring_layout(G)
    nx.draw(G, pos=layout, with_labels=True, hold=False)
    # for index in G.edges_iter(data=True):
    #     print(index)   #输出所有边的节点关系和权重
    plt.show()

    # 最Naive的pagerank计算，最朴素的方式没有设置随机跳跃的部分，所以alpha=1
    pr_value = nx.pagerank(G, alpha=1)
    print("naive pagerank值是：", pr_value)

    # 改进后的pagerank计算，随机跳跃概率为15%，因此alpha=0.85
    pr_impro_value = nx.pagerank(G, alpha=0.85)
    print("improved pagerank值是：", pr_impro_value)

