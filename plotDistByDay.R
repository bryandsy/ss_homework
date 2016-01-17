answerPostByDay <- read.csv("PostTimeDist//answerPostByDay", header = FALSE, sep=' ')
questionPostByDay <- read.csv("PostTimeDist//questionPostByDay", header = FALSE, sep=' ')
commentDataByDay <- read.csv("PostTimeDist//commentDataByDay", header = FALSE, sep=' ')
voteTimeNegativeByDay <- read.csv("PostTimeDist//voteTimeNegativeByDay", header = FALSE, sep=' ')
voteTimePositiveByDay <- read.csv("PostTimeDist//voteTimePositiveByDay", header = FALSE, sep=' ')

#排除第一列（行名称）
result = (  questionPostByDay[, 2:ncol(questionPostByDay) ] * 0.43 
            + answerPostByDay[, 2:ncol(answerPostByDay) ] * 0.43 
            + commentDataByDay[, 2:ncol(commentDataByDay) ] * 0.14
         ) * 0.75
       + (  voteTimeNegativeByDay[, 2:ncol(voteTimeNegativeByDay)] * 0.25
            + voteTimeNegativeByDay[, 2:ncol(voteTimePositiveByDay)] * 0.75
         ) * 0.25
        

#行名称
row.names(result) <- answerPostByDay[,1]

#添加列名
colnames(result) <- as.character(seq(1,31))

remove(answerPostByDay)
remove(questionPostByDay)
remove(commentDataByDay)
remove(voteTimeNegativeByDay)
remove(voteTimePositiveByDay)

jpeg(filename = paste('Rplot08-15Day.jpg', sep=''), width = 1920, height = 1080, quality = 100)
#出图
heatmap.2(data.matrix(result), Rowv=NA, Colv=NA,
          key=T,keysize=1, trace="none",
          col=brewer.pal(9, "Blues"),
          #col=topo.colors(100),
          scale="none", 
          cexCol=2,cexRow=0.5,
          xlab = 'day', ylab = 'month')
dev.off()