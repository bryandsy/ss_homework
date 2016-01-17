# -*- coding: utf-8 -*-
'''
Created on Tue Jan 12 14:44:08 2016

@author: Jonater
'''
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
import sklearn.preprocessing as preprocessing

from sklearn.datasets import make_classification
from pandas import DataFrame
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from plot_learning_curve import plot_learning_curve 

mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

features = []
labels = []
def parseTestData(fn):
    with open(fn, 'rb') as inf:
        ll = inf.readlines()
        random.shuffle(ll, random.random)
        for l in ll:
            tmpd = eval(l.strip())
            features.append(tmpd['feature'])
            labels.append(tmpd['label'])
    return np.array(features), np.array(labels)

def algomain(df):
    scaler = preprocessing.StandardScaler()
    
    #liNum 只看>=2的部分
    df['liG2'] = (df.liNum > 2).astype(int)    
    
    #开头有Wh/Who且结尾有Q
    df['WhAndQ1'] = ((df.startWithWh == 1) & (df.endWithQ == 1)).astype(int)
    df['WhAndQ0'] = ((df.startWithWh == 0) & (df.endWithQ == 0)).astype(int)
    
    #标准化
    popTagsNum_scale_param = scaler.fit(df['popTagsNum'])
    df['popTagsNum_scaled'] = scaler.fit_transform(df['popTagsNum'], popTagsNum_scale_param)

    codeFragNum_scale_param = scaler.fit(df['codeFragNum'])
    df['codeFragNum_scaled'] = scaler.fit_transform(df['codeFragNum'], codeFragNum_scale_param)
    
    avgTI_scale_param = scaler.fit(df['avgTI'])
    df['avgTI_scaled'] = scaler.fit_transform(df['avgTI'], avgTI_scale_param)
    
    totalTI_scale_param = scaler.fit(df['totalTI'])
    df['totalTI_scaled'] = scaler.fit_transform(df['totalTI'], totalTI_scale_param)
    
    title_scale_param = scaler.fit(df['titleLength'])
    df['title_scaled'] = scaler.fit_transform(df['titleLength'], title_scale_param)
    
    body_scale_param = scaler.fit(df['bodyLength'])
    df['body_scaled'] = scaler.fit_transform(df['bodyLength'], body_scale_param)
    
    a_scale_param = scaler.fit(df['aNum'])
    df['a_scaled'] = scaler.fit_transform(df['aNum'], a_scale_param)
    
    strong_scale_param = scaler.fit(df['strongNum'])
    df['strong_scaled'] = scaler.fit_transform(df['strongNum'], strong_scale_param)
    
    thx_scale_param = scaler.fit(df['thxNum'])
    df['thx_scaled'] = scaler.fit_transform(df['thxNum'], thx_scale_param)
    
    dayhot_scale_param = scaler.fit(df['dayHot'])
    df['dayHot_scaled'] = scaler.fit_transform(df['dayHot'], dayhot_scale_param)    

    train_df = df[['class', 
                   'codeFragNum_scaled', 'liNum',
                   'totalTI', 'avgTI',
                   'popTagsNum_scaled', 
                   'startWithWh', 'endWithQ', 
                   'WhAndQ1', 'WhAndQ0',  'isweekend',
                   'cntQ', 'cntA',
                   'body_scaled', 'title_scaled',
                   'a_scaled', 'strong_scaled', 'thx_scaled', 'dayHot_scaled']]


    train_np = train_df.as_matrix()
    tX, ty = train_np[:, 1:], train_np[:, 0]

#    estm = LinearSVC(C=0.3, penalty='l1', dual=False)
    estm = SVC(C=0.1, kernel='linear')
    
    plot_learning_curve(estm, 'LinearSVC',
                        tX, ty, ylim=(0.5, 1.0), 
                        train_sizes=np.linspace(.1, 1, 10))
                    
    estm.fit(tX, ty)   
    print pd.DataFrame({'columns': list(train_df.columns[1:]), 
                        'coef': list(estm.coef_.T)})



def batchPredict(X, y):
    est = SGDClassifier(loss='log', penalty='l1', alpha=0.01)
    progressive_validation_score = []
    train_score = []
    l = len(X)
    step = 500
    for datapoint in range(0, l, step):
        X_batch = X[datapoint:datapoint+step]
        y_batch = y[datapoint:datapoint+step]
        if datapoint > 0:
            progressive_validation_score.append(est.score(X_batch, y_batch))
        est.partial_fit(X_batch, y_batch, classes=range(10))
        if datapoint > 0:
            train_score.append(est.score(X_batch, y_batch))
    
    plt.plot(train_score, label='train score')
    plt.plot(progressive_validation_score, label='progressive validation score')
    plt.xlabel('Mini-batch')
    plt.ylabel('Score')
    plt.legend(loc='best')  
    plt.show()       



def pltDataFrame(df):
    fig, axes = plt.subplots(nrows=2, ncols=4)
    fig.set(alpha=0.2)

    ans1oncntQ = df.cntQ[df['class'] == 1].value_counts()
    ans0oncntQ = df.cntQ[df['class'] == 0].value_counts()
    DataFrame({u'回答':ans1oncntQ,
               u'未回答':ans0oncntQ}) \
    .plot(kind='kde', stacked=False, 
          ax=plt.subplot2grid((2,4),(0,0), colspan=2))
    
    plt.title(u'按用户发帖看回答情况')
    plt.ylabel(u'密度')
    plt.xlabel(u'提问数量')
    
#    ans1oncntA = df.cntA[df['class'] == 1].value_counts()
#    ans0oncntA = df.cntA[df['class'] == 0].value_counts()
#    DataFrame({u'回答':ans1oncntA,
#               u'未回答':ans0oncntA}) \
#    .plot(kind='kde', stacked=False, 
#          ax=plt.subplot2grid((2,4),(0,2), colspan=2))
#    
#    plt.title(u'按用户回帖看回答情况')
#    plt.ylabel(u'密度')
#    plt.xlabel(u'提问数量')
    
#    ans1onwkd = df.isweekend[df['class'] == 1].value_counts()
#    ans0onwkd = df.isweekend[df['class'] == 0].value_counts()
#    DataFrame({u'回答':ans1onwkd,
#               u'未回答':ans0onwkd}) \
#    .plot(kind='bar', stacked=False, 
#          ax=plt.subplot2grid((2,4),(0,0)))
#    plt.title(u'按周末看回答情况')
#    plt.ylabel(u'数量')
#    plt.xlabel(u'是否周末')

#    ans1onli = df.liNum[df['class'] == 1].value_counts()
#    ans0onli = df.liNum[df['class'] == 0].value_counts()
#    DataFrame({u'回答':ans1onli,
#               u'未回答':ans0onli}) \
#    .plot(kind='bar', stacked=False, 
#          ax=plt.subplot2grid((2,4),(0,0), colspan=2))
#    
#    plt.title(u'按li数量看回答情况')
#    plt.ylabel(u'数量')
#    plt.xlabel(u'li数量')
    


#    ans1onCodeFregNum = df.codeFragNum[df['class'] == 1].value_counts()
#    ans0onCodeFregNum = df.codeFragNum[df['class'] == 0].value_counts()
#    DataFrame({u'回答':ans1onCodeFregNum,
#               u'未回答':ans0onCodeFregNum}) \
#    .plot(kind='kde', stacked=False, 
#          ax=plt.subplot2grid((2,4),(0,0), colspan=2))
#    
#    plt.title(u'按code数量看回答情况')
#    plt.ylabel(u'密度')
#    plt.xlabel(u'code数量')
    

#    plt.subplot2grid((2,4),(0,2), colspan=2)
#    df.avgTI[df['class'] == 0].plot(kind='kde')    
#    df.avgTI[df['class'] == 1].plot(kind='kde')
#    plt.title(u'按平均流行Tag指数看回答情况')
#    plt.ylabel(u'密度')
#    plt.xlabel(u'平均流行tag指数')
#    plt.legend((u'未回答', u'回答'), loc='best')


#    
#    ans0onWh = df.startWithWh[df['class'] == 0].value_counts()
#    ans1onWh = df.startWithWh[df['class'] == 1].value_counts()
#    DataFrame({u'回答':ans1onWh,
#               u'未回答':ans0onWh}) \
#    .plot(kind='bar', stacked=False,
#          ax=plt.subplot2grid((2,4),(1,2)))
#    plt.title(u'按标题是否以Wh开头看回答情况')
#    plt.ylabel(u'数量')
#    
#    ans0onQ = df.endWithQ[df['class'] == 0].value_counts()
#    ans1onQ = df.endWithQ[df['class'] == 1].value_counts()
#    DataFrame({u'回答':ans1onQ,
#               u'未回答':ans0onQ}) \
#    .plot(kind='bar', stacked=False,
#          ax=plt.subplot2grid((2,4),(1,3)))
#    plt.title(u'按标题是否以Q结尾看回答情况')
#    plt.ylabel(u'数量')    

        
#    plt.subplot2grid((2,4),(1,0), colspan=2)
#    df.bodyLength[df['class'] == 0].plot(kind='kde')
#    df.bodyLength[df['class'] == 1].plot(kind='kde')
#    
#    plt.title(u'按内容长度看回答情况')
#    plt.ylabel(u'密度')
#    plt.xlabel(u'长度')
#    plt.legend((u'未回答', u'回答'), loc='best')
#    
#    
#    plt.subplot2grid((2,4),(1,2), colspan=2)
#    df.titleLength[df['class'] == 0].plot(kind='kde')
#    df.titleLength[df['class'] == 1].plot(kind='kde')
#    
#    plt.title(u'按标题长度看回答情况')
#    plt.ylabel(u'密度')
#    plt.xlabel(u'长度')
#    plt.legend((u'未回答', u'回答'), loc='best')
    
    
    plt.show()
    
def plotpair(df):
#    pr = sns.pairplot(df[:500], vars=['codeFragNum', 'liNum', 'popTagsNum',
#                                      'bodyLength', 'titleLength'],
#                      hue='class', size=1.5)
    cor = sns.corrplot(df, annot=False)    
    
if __name__ == '__main__':
    X, y = parseTestData('testData.txt')
    featureName= ['codeFragNum', 'liNum', 'popTagsNum',
                  'totalTI', 'avgTI','bodyLength', 'titleLength',
                   'startWithWh', 'endWithQ', 'isweekend',
                   'cntQ', 'cntA', 'aNum', 'strongNum', 'thxNum', 'dayHot']
    df = DataFrame(np.hstack((X, y[:, None])), columns=featureName +['class'])
    
#    pltDataFrame(df)
#    plotpair(df)
    algomain(df)
        