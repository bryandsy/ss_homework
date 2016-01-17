# coding:utf-8
import sys
import xml.sax
from urllib import quote_plus

class StackContentHandler(xml.sax.ContentHandler):

    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
 
    def startElement(self, name, attrs):
        questionId = "1"
        answerId = "2"
        
        # only 'row' elements are relevant, skip elements that are not rows
        if name != "row":
            return

        ptype = "__none__"
        if attrs.has_key("PostTypeId"):
            ptype = attrs.getValue("PostTypeId")

        try:
            if ptype == questionId:
                line = self.getQuestion(attrs)
                if line: questionFile.write(line +'\n')
            elif ptype == answerId:
                line = self.getAnswer(attrs)
                if line: answerFile.write(line +'\n')
        except:
            pass
            
    def getAnswer(self, attrs):
        #回答ID
        id = "__none__"
        if attrs.has_key("Id"):
            id = attrs.getValue("Id")
        
        #回答的问题ID
        parentId = "__none__"
        if attrs.has_key("ParentId"):
            parentId = attrs.getValue("ParentId")
            
        #回答创建时间
        creationDate = "__none__"
        if attrs.has_key("CreationDate"):
            creationDate = attrs.getValue("CreationDate")
        
        #回答被赞数
        score = "0"
        if attrs.has_key("Score"):
            score = attrs.getValue("Score")
            
        #回答内容
        body = "__none__"
        if attrs.has_key("Body"):
            body = attrs.getValue("Body")
        body = quote_plus(body)
        
        #回答人ID
        owner = "__none__"
        if attrs.has_key("OwnerUserId"):
            owner = attrs.getValue("OwnerUserId")
        
        #回答评论数量
        commentCount = "0"
        if attrs.has_key("CommentCount"):
            commentCount = attrs.getValue("CommentCount")
            
        line = []
        line.append(id)
        line.append(parentId)
        line.append(creationDate)
        line.append(score)
        line.append(body)
        line.append(owner)
        line.append(commentCount)
        
        return "|".join(line)
        
    def getQuestion(self, attrs):
        #问题ID
        id = "__none__"
        if attrs.has_key("Id"):
            id = attrs.getValue("Id")
            
        #接受的回答ID
        acceptedAnswerId = "__none__"
        if attrs.has_key("AcceptedAnswerId"):
            acceptedAnswerId = attrs.getValue("AcceptedAnswerId")
            
        #问题创建时间
        creationDate = "__none__"
        if attrs.has_key("CreationDate"):
            creationDate = attrs.getValue("CreationDate")
        
        #问题被赞数
        score = "0"
        if attrs.has_key("Score"):
            score = attrs.getValue("Score")
           
        #查看数量
        viewCount = "0"
        if attrs.has_key("ViewCount"):
            viewCount = attrs.getValue("ViewCount")

        #帖子内容
        body = "__none__"
        if attrs.has_key("Body"):
            body = attrs.getValue("Body")
        body = quote_plus(body)
        
        #提问人ID
        owner = "__none__"
        if attrs.has_key("OwnerUserId"):
            owner = attrs.getValue("OwnerUserId")
        
        #帖子标题
        title = "__none__"
        if attrs.has_key("Title"):
            title = attrs.getValue("Title")
    
        #问题标签
        tags = "__none__"
        if attrs.has_key("Tags"):
            tags = attrs.getValue("Tags")
        
        #回答的数量
        answerCount = "0"
        if attrs.has_key("AnswerCount"):
            answerCount = attrs.getValue("AnswerCount")
            
        #评论数量
        commentCount = "0"
        if attrs.has_key("CommentCount"):
            commentCount = attrs.getValue("CommentCount")
            
        #收藏数量
        favoriteCount = "0"
        if attrs.has_key("FavoriteCount"):
            favoriteCount = attrs.getValue("FavoriteCount")
        
            
        line = []
        line.append(id)
        line.append(acceptedAnswerId)
        line.append(creationDate)
        line.append(score)
        line.append(viewCount)
        line.append(body)
        line.append(owner)
        line.append(title)
        line.append(tags)
        line.append(answerCount)
        line.append(commentCount)
        line.append(favoriteCount)
        
        return "|".join(line)
 
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: " + sys.argv[0] + " <file>"
        sys.exit(1)

    fname = sys.argv[1]
    #print fname

    f = open(fname)
    questionFile = open('Question.txt','wb')
    answerFile = open('Answer.txt', 'wb')
    xml.sax.parse(f, StackContentHandler())
    questionFile.close()
    answerFile.close()
    f.close()

