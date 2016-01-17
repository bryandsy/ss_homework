answerPostByHour <- read.csv("PostTimeDist//answerPostByHour", header = FALSE, sep=' ')
questionPostByHour <- read.csv("PostTimeDist//questionPostByHour", header = FALSE, sep=' ')
commentDataByHour <- read.csv("PostTimeDist//commentDataByHour", header = FALSE, sep=' ')

#排除第一列（行名称）
result = questionPostByHour[, 2:ncol(questionPostByHour) ] * 0.43 
         + answerPostByHour[, 2:ncol(answerPostByHour) ] * 0.43 
         + commentDataByHour[, 2:ncol(commentDataByHour) ] * 0.14

#行名称
rownames(result) <- answerPostByHour[,1]

#添加列名
colnames(result) <- as.character(seq(0,23))

remove(answerPostByHour)
remove(questionPostByHour)
remove(commentDataByHour)

jpeg(filename = paste('Rplot08-15Hour.jpg', sep=''), width = 1920, height = 1080, quality = 100)

#出图
heatmap.2(data.matrix(result), Rowv=NULL, Colv=NULL,
          key=T,keysize=1.5, trace="none",
          col=brewer.pal(9, "Blues"),
          #col=topo.colors(100),
          scale="none",
          cexCol=2, cexRow=0.5,
          #labRow=NA,
          xlab = 'hour', ylab = 'day')
dev.off()
