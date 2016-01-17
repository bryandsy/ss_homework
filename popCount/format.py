#coding:utf-8

language = ['c', 'c++', 'c#', 'objective-c', 'java', 'javascript',  'python', 'php', 'css']
d = {i:{y:0 for y in xrange(2009,2016)} for i in language}

with open("sortPopCount.txt") as inf:
    for i in inf:
        lang, year, cnt = i.strip().split(' ')
        year, cnt = int(year), int(cnt)
        d[lang][year] = cnt
        
        
with open('result.txt', 'wb') as outf:
    for year in xrange(2009, 2016):
        outf.write('%d [' % year)
        for lang in language:
            outf.write(str(d[lang][year]))
            print lang, year
            if lang is not 'css':
                outf.write(',')
        outf.write(']\n')
