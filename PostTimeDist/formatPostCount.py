#coding:utf-8

import re
from multiprocessing.dummy import Pool as ThreadPool

HourRegex = re.compile(r"\(u?'(\d{4})-(\d{2})-(\d{2})T(\d{2})', (\d+)\)")
DayRegex = re.compile(r"\(u?'(\d{4})-(\d{2})-(\d{2})', (\d+)\)")

def initDic(year, isHour=False):
    #初始化计数字典
    m31 = [1, 3, 5, 7, 8, 10, 12]
    
    dic = {m:{} for m in xrange(1,13)}
    for m in xrange(1,13):
        if m in m31:
            dd = 31
        elif m == 2:
            dd = 28
            if (year%4 == 0 and (year%100 != 0 or year%400 == 0)):
                dd = 29
        else:
            dd = 30

        if isHour:
            dic[m] = {d:{h:0 for h in xrange(24)} for d in xrange(1, dd+1)}
        else:
            dic[m] = {d:0 for d in xrange(1, dd+1)}
    
    return dic

def CountByHour(outFileName, regex):
    outFile = open(outFileName, 'wb')
    for i in xrange(0,8):
        inFileName = 'PostTimeDist/' +outFileName +'/part-0000' +str(i)
        with open(inFileName, 'rb') as inf:
            #取首行获取年份
            line = inf.readline()
            try:
                year, month, day, hour, cnt = map(lambda _:int(_), regex.findall(line)[0])
            except Exception, e:
                print e, inFileName, line
                year = 2008 + i
               
            dic = initDic(year, isHour=True)
            dic[month][day][hour] += cnt
            
            for line in inf:
                try:
                    _, month, day, hour, cnt = map(lambda _:int(_), regex.findall(line)[0])
                    dic[month][day][hour] += cnt
                except Exception, e:
                    print e, inFileName, line

            for m in xrange(1,13):
                for d in xrange(1, dic[m].__len__()+1):
                    outFile.write('%d-%d-%d' % (year, m, d))
                    for h in xrange(24):
                        outFile.write(' %d' % dic[m][d][h])
                    outFile.write('\n')
    outFile.close()

def CountByDay(outFileName, regex):
    outFile = open(outFileName, 'wb')
    for i in xrange(0,8):
        inFileName = 'PostTimeDist/' +outFileName +'/part-0000' +str(i)
        with open(inFileName, 'rb') as inf:
            #取首行获取年份
            line = inf.readline()
            try:
                year, month, day, cnt = map(lambda _:int(_), regex.findall(line)[0])
            except Exception, e:
                print e, inFileName, line
                year = 2008 + i

            dic = initDic(year, isHour=False)
            dic[month][day] += cnt
            
            for line in inf:
                try:
                    _, month, day, cnt = map(lambda _:int(_), regex.findall(line)[0])
                    dic[month][day] += cnt
                except Exception, e:
                    print e, inFileName, line

            for m in xrange(1,13):
                outFile.write('%d-%d' % (year, m))
                for d in xrange(1, 32):
                    if dic[m].has_key(d):
                        outFile.write(' %d' % dic[m][d])
                    else:
                        outFile.write(' 0')
                outFile.write('\n')
    outFile.close()
def run(FileName):
    if FileName.endswith('ByHour'):
        CountByHour(FileName, HourRegex)
    else:
        CountByDay(FileName, DayRegex)

def main():
    targetList = []
    tableName = ['questionPost', 'answerPost', 'commentData', 'voteTimeNegative', 'voteTimePositive']
    for tn in tableName:
        for tnsuffix in ['ByHour', 'ByDay']:
            if tn.startswith('voteTime') and tnsuffix is 'ByHour': continue  #jump voteTime count by Hour
            targetList.append(tn +tnsuffix)
    
    pool = ThreadPool(5)
    pool.map(run, targetList)
    pool.close()
    pool.join()
    
main()
