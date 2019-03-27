# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/4/2 16:47
'''
function:
    transfrom data to graph to observe the distribution characteristics.
'''
import matplotlib.pyplot as plt
import numpy as np

def readData(filepath):
    '''
    :param: filepath: the path of samples
    :return: iris_matrix_X： a matrix of attributes
             iris_array_y: an array of label
    '''
    iris_matrix_X = np.zeros(150*4).reshape((150, 4))   # feature data
    iris_array_y = np.zeros(150)   # label, an array
    with open(filepath, encoding='UTF-8') as f:
        for i in range(0, 150):
            row = f.readline()
            elememts = row.split(",")    # split the 5 elements of one row
            iris_matrix_X[i, :] = [float(elememts[index]) for index in range(0, 4)]
            map = {"Iris-setosa": 1, "Iris-versicolor": 2, "Iris-virginica": 3}
            iris_array_y[i] = map.get(elememts[4].strip())   # get rid of the '\n' after iris label.
    return iris_matrix_X, iris_array_y

filepath = './data/bezdekIris.data.txt'
iris_X, iris_y = readData(filepath)

plt.subplot(441)    # 表示把画布分成4行4列，后面第三个数字范围可以是1~16
plt.ylabel('petal width')
plt.xticks([])
plt.yticks([0.0, 2.0])
plt.scatter(iris_X[:, 0], iris_X[:, 3], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(442)
plt.xticks([])
plt.yticks([])
plt.scatter(iris_X[:, 1], iris_X[:, 3], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(443)
plt.xticks([])
plt.yticks([])
plt.scatter(iris_X[:, 2], iris_X[:, 3], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(444)
plt.xticks([])
plt.yticks([])
plt.scatter(iris_X[:, 3], iris_X[:, 3], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(445)
plt.xticks([])
plt.ylabel('petal length')
plt.scatter(iris_X[:, 0], iris_X[:, 2], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(446)
plt.xticks([])
plt.yticks([])
plt.scatter(iris_X[:, 1], iris_X[:, 2], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(447)
plt.xticks([])
plt.yticks([])
plt.scatter(iris_X[:, 2], iris_X[:, 2], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(448)
plt.xticks([])
plt.yticks([])
plt.scatter(iris_X[:, 3], iris_X[:, 2], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(449)
plt.xticks([])
plt.yticks([2.0, 4.0])
plt.ylabel('sepal width')
plt.scatter(iris_X[:, 0], iris_X[:, 1], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(4, 4, 10)
plt.xticks([])
plt.yticks([])
plt.scatter(iris_X[:, 1], iris_X[:, 1], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(4, 4, 11)
plt.xticks([])
plt.yticks([])
plt.scatter(iris_X[:, 2], iris_X[:, 1], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(4, 4, 12)
plt.xticks([])
plt.yticks([])
plt.scatter(iris_X[:, 3], iris_X[:, 1], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(4, 4, 13)
plt.xlabel('sepal length')
plt.ylabel('sepal length')
plt.xticks([6.0, 8.0])
plt.scatter(iris_X[:, 0], iris_X[:, 0], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(4, 4, 14)
plt.yticks([])
plt.xticks([2.0, 4.0])
plt.xlabel('sepal width')
plt.scatter(iris_X[:, 1], iris_X[:, 0], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(4, 4, 15)
plt.yticks([])
plt.xlabel('petal length')
plt.scatter(iris_X[:, 2], iris_X[:, 0], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.subplot(4, 4, 16)
plt.yticks([])
plt.xticks([0.0, 2.0])
plt.xlabel('petal width')
plt.scatter(iris_X[:, 3], iris_X[:, 0], c=iris_y, marker='o', s=10, cmap=plt.cm.Spectral)

plt.show()
