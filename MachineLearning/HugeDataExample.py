# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:26:59 2016

@author: Jonater
"""
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

#生成大样本，高纬度特征数据
X, y = make_classification(200000, n_features=200, n_informative=25, n_redundant=0, n_classes=10, class_sep=2, random_state=0)

#用SGDClassifier做训练，并画出batch在训练前后的得分差
from sklearn.linear_model import SGDClassifier
est = SGDClassifier(loss="log", penalty="l1", alpha=0.001)
progressive_validation_score = []
train_score = []
for datapoint in range(0, 199000, 1000):
    X_batch = X[datapoint:datapoint+1000]
    y_batch = y[datapoint:datapoint+1000]
    if datapoint > 0:
        progressive_validation_score.append(est.score(X_batch, y_batch))
    est.partial_fit(X_batch, y_batch, classes=range(10))
    if datapoint > 0:
        train_score.append(est.score(X_batch, y_batch))

plt.plot(train_score, label="train score")
plt.plot(progressive_validation_score, label="progressive validation score")
plt.xlabel("Mini-batch")
plt.ylabel("Score")
plt.legend(loc='best')  
plt.show()               