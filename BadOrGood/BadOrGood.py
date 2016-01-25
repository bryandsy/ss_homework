# coding:utf-8
#bin/spark-submit 
import sys
import re
import subprocess

from pyspark import SparkContext,SparkConf
from pyspark.sql import * 
from pyspark.sql.functions import *

from operator import add
from math import log
from random import randint
from datetime import datetime

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.linalg import Vectors, DenseVector
from pyspark.mllib.tree import DecisionTree, RandomForest

from otherDic import *

def extractFeature(dfrow):
    #dfrow contains id, date, body, title, tags, status, cntQ, cntA
    codeFragmentNum = dfrow.body.count('%3Ccode%3E') 
    liNum = dfrow.body.count('%3Cli%3E')
    aNum = dfrow.body.count('%3Ca+href%3D')
    strongNum = dfrow.body.count('%3Cstrong%3E')
    thxNum = dfrow.body.count('Thank') + dfrow.body.count('thank')
    
    useTags = map( lambda _: _.strip('<>'), filter( lambda _: _, dfrow.tags.split('<')))

    usePopTags = list( set( popTags.keys() ).__and__( set( useTags ) ) )          
    popTagsNum = len(usePopTags)

    tmpcnt = 0.0
    for k in useTags:
        if popTags.has_key(k): tmpcnt += popTags[k]
        else: tmpcnt += randint(100, 500)
    totalTI = tmpcnt # TI stands for Tags Index
    avgTI = totalTI / len(useTags) if len(useTags) else totalTI
    
    bodyLength = len(dfrow.body)
    titleLength = len(dfrow.title)

    istitleBeginWithWh = 1 if dfrow.title.startswith('Wh') or dfrow.title.startswith('How') else 0
    istitleEndWithQ = 1 if dfrow.title.endswith('?') else 0
    istitleWhAndQ1 = istitleBeginWithWh & istitleEndWithQ
    istitleWhAndQ0 = 0 if (istitleBeginWithWh | istitleEndWithQ) else 1
    
    isweekend = 1 if datetime.strptime(dfrow.date.split('T')[0], '%Y-%m-%d').isoweekday() > 5 else 0

    ymd, hms = dfrow.date.split('T')
    year, month, day = map(lambda _:int(_), ymd.split('-'))
    hour, _ = hms.split(':', 1)
    hour = int(hour)
    hourHot = hourhotDic[year][month][day][hour]
    
    try:
        cntQ = int(dfrow.cntQ)
    except:
        cntQ = 0

    try:
        cntA = int(dfrow.cntA)
    except:
        cntA = 0

    featureList = []
    featureList.append(codeFragmentNum)
    featureList.append(liNum)
    featureList.append(popTagsNum)
    featureList.append(totalTI)
    featureList.append(avgTI)
    featureList.append(bodyLength)
    featureList.append(titleLength)
    featureList.append(istitleBeginWithWh)
    featureList.append(istitleEndWithQ)
    featureList.append(istitleWhAndQ1)
    featureList.append(istitleWhAndQ0)
    featureList.append(isweekend)
    featureList.append(cntQ)
    featureList.append(cntA)
    featureList.append(aNum)
    featureList.append(strongNum)
    featureList.append(thxNum)
    featureList.append(hourHot)
    return featureList

def fetchDataToFile(hc, filename):
    subprocess.call(["hadoop", "dfs", "-rm", "-r", "-f", filename])

    ucntQ = hc.sql("select owner as qowner, count(1) as cntQ from questionPost where owner != '__none__' group by owner")
    
    ucntA = hc.sql("select owner as aowner, count(1) as cntA from answerPost where owner != '__none__' group by owner ")

    Positivedf = hc.sql("select id, owner, creationDate as date, body, title, tags, 1 as status from questionPost " \
                        "where substr(creationDate,0,4) > '2013' " \
                        "and substr(creationDate,0,4) != '2016' " \
                        "and answerCount > 4 " \
                        "limit 30000") 
                       
    Negativedf = hc.sql("select id, owner, creationDate as date, body, title, tags, 0 as status from questionPost " \
                        "where substr(creationDate,0,4) = '2015' " \
                        "and answerCount  = 0 " \
                        "and commentCount = 0 " \
                        "and body not like '%Possible+Duplicate%' " \
                        "limit 30000")
    
    Posdf = Positivedf.join(ucntQ, col('owner') == col('qowner'), 'inner').drop('qowner') \
                      .join(ucntA, col('owner') == col('aowner'), 'inner').drop('aowner')
    Negdf = Negativedf.join(ucntQ, col('owner') == col('qowner'), 'inner').drop('qowner') \
                      .join(ucntA, col('owner') == col('aowner'), 'inner').drop('aowner')

    AllDataRowrdd = Posdf.rdd.union(Negdf.rdd)
    AllDataRowrdd.repartition(10).saveAsPickleFile(filename)

def fone(rdd):
    # rdd with the format {'label':1, 'predict':1}
    precision = float(rdd.filter(lambda _: _['predict'] == 1 and _['label'] == 1).count()) / float(rdd.filter(lambda _: _['predict'] == 1).count())
    recall = float(rdd.filter(lambda _: _['predict'] == 1 and _['label'] == 1).count()) / float(rdd.filter(lambda _: _['label'] == 1).count())
    return 2 * ( precision * recall ) / (precision + recall)

def accuracy(rdd):
    # rdd with the format {'label':1, 'predict':1}
    return float(rdd.filter(lambda _: _['label'] == _['predict']).count()) / float(rdd.count())

def test(model, testrdd):
    #testrdd with the format {'label':1, 'feature': [a,b,c]}
    predictrdd = model.predict(testrdd.map(lambda _: _['feature']))
    resultrdd = testrdd \
                .map(lambda _: _['label']) \
                .zip(predictrdd) \
                .map(lambda _: {'label': _[0], 'predict': _[1]})
    return resultrdd

def main():
    appName = "BadOrGood;zl"
    
    conf = (SparkConf()
            .setAppName(appName)
            .set("spark.executor.memory", "5g")
            .set("spark.executor.cores","3")
            .set("spark.executor.instance", "3")
            )
    sc = SparkContext(conf = conf)
    hc = HiveContext(sc)

    #fetch data
    #filepath = '/sshomework_zl/BadOrGood/AllDataRowrdd'
    #fetchDataToFile(hc, filepath)
    
    #load data
    # AllDataRawrdd = sc.pickleFile(filepath) \
                    # .map( lambda _: {'label':int(_.status), 'feature':extractFeature(_)} ) \
                    # .repartition(10)
    
    AllDataRawrdd = sc.pickleFile('/pickleData').repartition(10)
    
    
    #standardizer for train and test data
    model = StandardScaler(True, True) \
            .fit( AllDataRawrdd \
                  .map( lambda _: Vectors.dense(_['feature']) ) 
            )
    labels = AllDataRawrdd.map(lambda _: _['label'])
    featureTransformed = model.transform( AllDataRawrdd.map(lambda _: _['feature']) )
    AllDataRawrdd = labels \
                    .zip(featureTransformed) \
                    .map( lambda _: { 'label':_[0], 'feature':_[1] } )
    #sampling
    trainDataRawrdd, testDataRawrdd = AllDataRawrdd.randomSplit(weights=[0.7, 0.3], seed=100)
    trainDatardd = trainDataRawrdd.map( lambda _: LabeledPoint( _['label'], _['feature'] ) ).persist()
    testDatardd = testDataRawrdd.map( lambda _: {'label': _['label'], 'feature': list(_['feature']) } ).persist()
    
    #prediction & test
    lrmLBFGS = LogisticRegressionWithLBFGS.train(trainDatardd, iterations=3000, regParam=0.01, regType="l1")
    resultrdd = test(lrmLBFGS, testDatardd)
    lrmLBFGSFone = fone(resultrdd)
    lrmLBFGSac = accuracy(resultrdd)

    lrmSGD = LogisticRegressionWithSGD.train(trainDatardd, iterations=3000, step=0.1, regParam=0.01, regType="l1")
    resultrdd = test(lrmSGD, testDatardd)
    lrmSGDFone = fone(resultrdd)
    lrmSGDac = accuracy(resultrdd)
  
    dt = DecisionTree.trainClassifier(trainDatardd, 2, {}, maxDepth=10)
    resultrdd = test(dt, testDatardd)
    dtFone = fone(resultrdd)
    dtac = accuracy(resultrdd)
  
    rf = RandomForest.trainClassifier(trainDatardd, 2, {}, 10)
    resultrdd = test(rf, testDatardd)
    rfFone = fone(resultrdd)
    rfac = accuracy(resultrdd)

    print "LR_LBFGS f1 is : %f, ac is : %f" % (lrmLBFGSFone, lrmLBFGSac)
    print "LR_SGD f1 is : %f, ac is : %f" % (lrmSGDFone, lrmSGDac)
    print "Decision Tree f1 is: %f, ac is : %f" % (dtFone, dtac)
    print "Random Forest f1 is: %f, ac is : %f" % (rfFone, rfac)

    print lrmLBFGS.weights
    print lrmSGD.weights

    sc.stop()

if __name__ == '__main__':
    main()
