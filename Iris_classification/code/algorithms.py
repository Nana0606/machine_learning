# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/4/2 15:22
'''
function:
    3 algorithms on Iris dataset, Logistic Regression, Decision Tree and MLP classification.
'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier


def read_data(filepath):
    '''
    read data and transform those data to matrix or array to use.
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

def standardization(X_train, X_test):
    # 分别初始化对特征和目标值的标准化器
    ss_X = StandardScaler()

    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)    # 使用同一套规则标准化X_test
    return X_train, X_test

def logistic_regression(iris_X, iris_y):
    '''
    :param iris_X: training dataset
    :param iris_y: testing dataset
    :return: evaluation_result, evaluation matrics, including precision, recall and f1-score
    '''
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.25, random_state=33)  # split training set and testing set
    # X_train, X_test = standardization(X_train, X_test)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)  # train data
    lr_y_predict = lr.predict(X_test)  # test samples
    evaluation_result = classification_report(lr_y_predict, y_test,
                                   target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])  # generating evaluation report automatically
    return evaluation_result

def decision_tree(iris_X, iris_y):
    '''
    :param iris_X: training dataset
    :param iris_y: testing dataset
    :return:  evaluation_result, evaluation matrics, including precision, recall and f1-score
    '''
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.25, random_state=33)  # split training set and testing set
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)    # train data
    tree_y_predict = tree.predict(X_test)   # test samples
    print("样本真实值标签为: ")
    print(y_test)
    print("样本预测值标签为: ")
    print(tree_y_predict)
    evaluation_result = classification_report(tree_y_predict, y_test,
                                   target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])   # generating evaluation report automatically
    return evaluation_result


def MLP_class(iris_X, iris_y):
    '''
    :param iris_X: training dataset
    :param irix_y: testing dataset
    :return: predication is the class of predicted result, proba is the predication probability,
    '''
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.25, random_state=33)  # split training set and testing set
    # solver is used to specify the optimization method.
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='relu', hidden_layer_sizes=(5, 3), random_state=1)
    clf.fit(X_train, y_train)
    clf_y_predict = clf.predict(X_test)   # predict category
    print(y_test)
    evaluation_result = classification_report(clf_y_predict, y_test,
                                   target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])  # generating evaluation report automatically
    return evaluation_result


filepath = '../data/bezdekIris.data.txt'
iris_X, iris_y = read_data(filepath)
evaluation_logistic = logistic_regression(iris_X, iris_y)
print("下面是Logistic回归的评估结果：**************************")
print(evaluation_logistic)
evaluation_tree = decision_tree(iris_X, iris_y)
print("下面是决策树的评估结果：**************************")
print(evaluation_tree)
evaluation_MLPclf = MLP_class(iris_X, iris_y)
print("下面是MLP分类器的评估结果：**************************")
print(evaluation_MLPclf)









