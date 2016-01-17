# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:23:06 2016

@author: Jonater
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from pandas import DataFrame
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from plot_learning_curve import plot_learning_curve 

X, y = make_classification(1000, n_features=10, n_informative=2, 
                           n_redundant=2, n_classes=2, random_state=0)

df = DataFrame(np.hstack((X, y[:, None])), columns=range(10) +["class"])

#pr = sns.pairplot(df[:50], vars=[2,3,6,8], hue="class", size=1.5)
#cor = sns.corrplot(df, annot=False)

#增强正则化作用，减小C值
#减少特征，指定feature绘制学习曲线
#plot_learning_curve(LinearSVC(C=0.5), "LinearSVC(C=0.1)",
#                    X[:,[2,3,6,8]], y, ylim=(0.6, 1.01),
#                    train_sizes=np.linspace(.3,1,6))

#自动挑选feature组合
#plot_learning_curve(Pipeline([("fs", SelectKBest(f_classif, k=2)), # select two features
#                            ("svc", LinearSVC(C=0.2))]), 
#                    "SelectKBest(f_classif, k=2) + LinearSVC(C=10.0)", 
#                    X, y, ylim=(0.8, 1.0), train_sizes=np.linspace(.1, 1, 5))

#自动调整正则化C值
#estm = GridSearchCV(LinearSVC(),
#                    param_grid={"C":[0.001, 0.01, 0.1, 1.0, 10.0]})
#plot_learning_curve(estm, "LinearSVC(C=AUTO)", 
#                    X, y, ylim=(0.8, 1.0), train_sizes=np.linspace(.1, 1, 5))
#print "Chosen parameter on 100 datapoints: %s" % estm.fit(X[:500], y[:500]).best_params_

#调整正则化l2 -> l1                    
#plot_learning_curve(LinearSVC(C=0.1, penalty='l1', dual=False), 
#                    "LinearSVC(C=0.1, penalty='l1')",
#                    X, y, ylim=(0.8, 1.0),
#                    train_sizes=np.linspace(.2, 1, 5))

#查看feature权重
#l2正则化，它对于最后的特征权重的影响是，尽量打散权重到每个特征维度上，不让权重集中在某些维度上，出现权重特别高的特征。
#而l1正则化，它对于最后的特征权重的影响是，让特征获得的权重稀疏化，也就是对结果影响不那么大的特征，干脆就拿不着权重。
#estm = LinearSVC(C=0.1, penalty='l2', dual=False)
#estm.fit(X, y)
#print "Coefficients learned: %s" % estm.coef_
#print "Non-zero coefficients: %s" % np.nonzero(estm.coef_)[1]



plt.show()


