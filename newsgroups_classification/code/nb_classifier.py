# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/6/10 20:11
from sklearn import datasets
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import MultinomialNB

"""
使用朴素贝叶斯进行文本分类
"""

rawData = datasets.load_files("../data/20_newsgroups/", encoding='latin-1')
print(len(rawData.data))     # 输出值为19997
print(rawData.target_names)

vec = CountVectorizer()    # 向量化模块
X = vec.fit_transform(rawData.data)
y = rawData.target
print("X shape is: ", X.shape)
print("y shape is: ", y.shape)

# print(MultinomialNB().get_params().keys())    # 查看MultinomialNB()可以使用的param_name


param_range = np.logspace(-6, -1, 5)
train_accuracy, validation_accuracy = validation_curve(MultinomialNB(), X, y, cv=5, param_name="alpha", param_range=param_range, scoring="accuracy")
print("train_score is: *************************************")
print(train_accuracy)
print("validation_score is: *************************************")
print(validation_accuracy)

train_accuracy_mean = train_accuracy.mean(1)
train_accuracy_std = train_accuracy.std(1)
validation_accuracy_mean = validation_accuracy.mean(1)
validation_accuracy_std = validation_accuracy.std(1)

print("validation_accuracy average: ", validation_accuracy_mean)

print("train_accuracy_mean: ", train_accuracy_mean.mean(0))
print("validation_accuracy_mean: ", validation_accuracy_mean.mean(0))

plt.semilogx(param_range, train_accuracy_mean, 'o-', label="train accuracy", color="r")
plt.semilogx(param_range, validation_accuracy_mean, 'o-', label="validation accuracy", color="g")
plt.legend(loc='best')
plt.show()


