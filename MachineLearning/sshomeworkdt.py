# -*- coding: utf-8 -*-
'''
Created on Tue Jan 12 14:44:08 2016

@author: Jonater
'''

import random
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import sklearn.preprocessing as preprocessing

from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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

    liNum_scale_param = scaler.fit(df['liNum'])
    df['liNum_scaled'] = scaler.fit_transform(df['liNum'], liNum_scale_param)
    
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
    
    hourHot_scale_param = scaler.fit(df['hourHot'])
    df['hourHot_scaled'] = scaler.fit_transform(df['hourHot'], hourHot_scale_param)    

    train_df = df[['class', 
                   'codeFragNum_scaled', 'liNum_scaled', 
                   'totalTI', 'avgTI',
                   'popTagsNum_scaled', 
                   'startWithWh', 'endWithQ', 
                   'WhAndQ1', 'WhAndQ0',  'isweekend',
                   'cntQ', 'cntA',
                   'body_scaled', 'title_scaled',
                   'a_scaled', 'strong_scaled', 'thx_scaled', 'hourHot_scaled']]
    
    train_np = train_df.as_matrix()
    tX, ty = train_np[:, 1:], train_np[:, 0]

    n_estimators = 800
    learning_rate = 0.8
    dt = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1)
    
    ada_real = AdaBoostClassifier(
                    base_estimator=dt,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    algorithm="SAMME.R")
                    

    plot_learning_curve(ada_real, 'AdaBoostWithDT',
                        tX, ty, ylim=(0.5, 1.0), 
                        cv=10, train_sizes=np.linspace(.1, 1, 10))
#    ada_real.fit(tX[:25000],ty[:25000])    
#    print ada_real.score(tX[25000:], ty[25000:])
    
    
if __name__ == '__main__':
    X, y = parseTestData('testData.txt')
    featureName= ['codeFragNum', 'liNum', 'popTagsNum',
                  'totalTI', 'avgTI','bodyLength', 'titleLength',
                   'startWithWh', 'endWithQ', 'isweekend',
                   'cntQ', 'cntA', 'aNum', 'strongNum', 'thxNum', 'hourHot']
    df = DataFrame(np.hstack((X, y[:, None])), columns=featureName +['class'])
    
    algomain(df)
        