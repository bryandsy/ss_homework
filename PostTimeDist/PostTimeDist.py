# coding:utf-8
#bin/spark-submit 
import sys

from pyspark import SparkContext,SparkConf
from pyspark.sql import * 
from pyspark.sql.functions import *

from operator import add

def main():
    appName = "PostTimeDist;zl"
    
    conf = (SparkConf()
            .setAppName(appName)
            .set("spark.executor.memory", "5g")
            .set("spark.executor.cores","3")
            .set("spark.executor.instance", "3")
            )
    sc = SparkContext(conf = conf)
    hc = HiveContext(sc)
    
    tableName = ['answerPost', 'questionPost', 'commentData']

    for tn in tableName:
        Daterdd = hc.sql("select creationDate as t from {tablename} where creationDate != '__none__' and creationDate not like '2016-%' ".format(tablename=tn)).rdd
        
        CountHourrdd = Daterdd \
                        .map(lambda _: ("{tillhour}".format(tillhour=_.t.split(':')[0]), 1)) \
                        .reduceByKey(add, 30)

        resultrdd = CountHourrdd \
                    .partitionBy(8, lambda k:int(k.split('-')[0])) # 2008-2015 --- 8 partitions, format: (date, count), date by hour
        resultrdd.saveAsTextFile('/sshomework_zl/PostTimeDist/%sByHour' % tn)

        CountDayrdd = CountHourrdd \
                         .map(lambda _:("{tillday}".format(tillday=_[0].split('T')[0]), _[1])) \
                         .reduceByKey(add, 30)

        resultrdd = CountDayrdd \
                    .partitionBy(8, lambda k:int(k.split('-')[0])) # same as ansCountHourrdd
        resultrdd.saveAsTextFile('/sshomework_zl/PostTimeDist/%sByDay' % tn)
    
    tn = 'voteTime' # votetime only could count by day
    Datedf = hc.sql("select status, substr(creationDate,0,10) as t, count(1) as c from %s where creationDate != '__none__' group by substr(creationDAte,0,10), status" % tn)
    Positiverdd = Datedf.where('status = 1').drop('status').rdd
    Negativerdd = Datedf.where('status = 0').drop('status').rdd

    PositiveCountDayrdd = Positiverdd.partitionBy(8, lambda k: int(k.split('-')[0]))
    NegativeCountDayrdd = Negativerdd.partitionBy(8, lambda k: int(k.split('-')[0]))

    PositiveCountDayrdd.saveAsTextFile('/sshomework_zl/PostTimeDist/voteTimePositiveByDay')
    NegativeCountDayrdd.saveAsTextFile('/sshomework_zl/PostTimeDist/voteTimeNegativeByDay')


    sc.stop()

if __name__ == '__main__':
    main()
