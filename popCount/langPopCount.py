# coding:utf-8
#bin/spark-submit 
import sys

from pyspark import SparkContext,SparkConf
from pyspark.sql import * 

def main():
    appName = "langPopCount;zl"
    
    conf = (SparkConf()
            .setAppName(appName)
            .set("spark.executor.memory", "5g")
            .set("spark.executor.cores","3")
            .set("spark.executor.instance", "3")
            )
    sc = SparkContext(conf = conf)
    hc = HiveContext(sc)

    langTagList = ['<java>', '<javascript>', '<c>', '<c++>', '<c#>', '<python>', '<php>', '<css>', '<html>', '<objective-c>']
    resultrdd = sc.emptyRDD()

    for tag in langTagList:
        postCountdf = hc.sql("select creationdate, 1 as c from questionpost where tags like '%{tag}%' ".format(tag=tag))
        postCountOnYearrdd = postCountdf \
                                 .filter(postCountdf.creationdate != '__none__') \
                                 .withColumn('year', postCountdf.creationdate.substr(0,4)) \
                                 .drop('creationdate') \
                                 .groupBy('year').count() \
                                 .withColumnRenamed('count', 'c') \
                                 .repartition(1) \
                                 .sort('year', ascending=True) \
                                 .map(lambda _: "{tag} {year} {cnt}".format(tag=tag.strip('<>'), year=_.year, cnt=_.c))
        resultrdd = resultrdd.union(postCountOnYearrdd)

    resultrdd = resultrdd.repartition(1)
    resultrdd.saveAsTextFile('/sshomework_zl/popCount')

    sc.stop()

if __name__ == '__main__':
    main()
