# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:44:08 2016

@author: Jonater
"""
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
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
    
    #标准化  
    popTagsNum_scale_param = scaler.fit(df['popTagsNum'])
    df['popTagsNum_scaled'] = scaler.fit_transform(df['popTagsNum'], popTagsNum_scale_param)

    liNum_scale_param = scaler.fit(df['liNum'])
    df['liNum_scaled'] = scaler.fit_transform(df['liNum'], liNum_scale_param)
    
    codeFragNum_scale_param = scaler.fit(df['codeFragNum'])
    df['codeFragNum_scaled'] = scaler.fit_transform(df['codeFragNum'], codeFragNum_scale_param)
    
    bodyLen_scale_param = scaler.fit(df['bodyLength'])
    df['bodyLen_scaled'] = scaler.fit_transform(df['bodyLength'], bodyLen_scale_param)

    titleLen_scale_param = scaler.fit(df['titleLength'])
    df['titleLen_scaled'] = scaler.fit_transform(df['titleLength'], titleLen_scale_param)

    train_df = df[['class', 
                   'codeFragNum_scaled', 'liNum_scaled', 'popTagsNum_scaled',
                   'startWithWh', 'endWithQ',
                   'bodyLen_scaled', 'titleLen_scaled']]
    train_np = train_df.as_matrix()
    
    tX = train_np[:, 1:]
    ty = train_np[:, 0]

    estm = SGDClassifier(loss='log', penalty='l1', alpha=0.015)
    plot_learning_curve(estm, "LogisticRegression(L1), cv=10-fold", 
                        tX, ty, ylim=(0.5, 1.0), 
                        cv=10, train_sizes=np.linspace(.1, 1, 10))
                    
    estm.fit(tX, ty)   
    print pd.DataFrame({'columns': list(train_df.columns[1:]), 
                        'coef': list(estm.coef_.T)})




def pltDataFrame(df):
    fig, axes = plt.subplots(nrows=2, ncols=4)
    fig.set(alpha=0.2)

    ans1onli = df.liNum[df["class"] == 1].value_counts()
    ans0onli = df.liNum[df["class"] == 0].value_counts()
    DataFrame({u'回答':ans1onli,
               u'未回答':ans0onli}) \
    .plot(kind='bar', stacked=False, 
          ax=plt.subplot2grid((2,4),(1,0), colspan=2))
    
    plt.title(u'按li数量看回答情况')
    plt.ylabel(u'数量')
    plt.xlabel(u'li数量')
    
#    plt.subplot2grid((2,4),(0,2), colspan=2)
#    df.avgTI[df["class"] == 0].plot(kind='kde')    
#    df.avgTI[df["class"] == 1].plot(kind='kde')
#    plt.title(u'按平均流行Tag指数看回答情况')
#    plt.ylabel(u'密度')
#    plt.xlabel(u'平均流行tag指数')
#    plt.legend((u'未回答', u'回答'), loc='best')

    
#    ans0onTags = df.popTagsNum[df["class"] == 0].value_counts()
#    ans1onTags = df.popTagsNum[df["class"] == 1].value_counts()
#    DataFrame({u'回答':ans1onTags,
#               u'未回答':ans0onTags}) \
#    .plot(kind='bar', stacked=False, 
#          ax=plt.subplot2grid((2,4),(1,0), colspan=2))
#    plt.title(u'按流行tag数量看回答情况')
    
    ans0onWh = df.startWithWh[df["class"] == 0].value_counts()
    ans1onWh = df.startWithWh[df["class"] == 1].value_counts()
    DataFrame({u'回答':ans1onWh,
               u'未回答':ans0onWh}) \
    .plot(kind='bar', stacked=False,
          ax=plt.subplot2grid((2,4),(1,2)))
    plt.title(u'按标题是否以Wh开头看回答情况')
    plt.ylabel(u'数量')
    
    ans0onQ = df.endWithQ[df["class"] == 0].value_counts()
    ans1onQ = df.endWithQ[df["class"] == 1].value_counts()
    DataFrame({u'回答':ans1onQ,
               u'未回答':ans0onQ}) \
    .plot(kind='bar', stacked=False,
          ax=plt.subplot2grid((2,4),(1,3)))
    plt.title(u'按标题是否以Q结尾看回答情况')
    plt.ylabel(u'数量')    

        
    plt.subplot2grid((2,4),(0,0), colspan=2)
    df.bodyLength[df["class"] == 0].plot(kind='kde')
    df.bodyLength[df["class"] == 1].plot(kind='kde')
    
    plt.title(u'按内容长度看回答情况')
    plt.ylabel(u'密度')
    plt.xlabel(u'长度')
    plt.legend((u'未回答', u'回答'), loc='best')
    
    
    plt.subplot2grid((2,4),(0,2), colspan=2)
    df.titleLength[df["class"] == 0].plot(kind='kde')
    df.titleLength[df["class"] == 1].plot(kind='kde')
    
    plt.title(u'按标题长度看回答情况')
    plt.ylabel(u'密度')
    plt.xlabel(u'长度')
    plt.legend((u'未回答', u'回答'), loc='best')
    
    
    plt.show()
if __name__ == '__main__':
    X, y = parseTestData('testData.txt')
    featureName= ['codeFragNum', 'liNum', 'popTagsNum',
                  'totalTI', 'avgTI','bodyLength', 'titleLength',
                   'startWithWh', 'endWithQ', 'isweekend',
                   'cntQ', 'cntA', 'aNum', 'strongNum', 'thxNum', 'dayHot']
    df = DataFrame(np.hstack((X, y[:, None])), columns=featureName +["class"])
    
#    pltDataFrame(df)
    algomain(df)
        