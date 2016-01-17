# coding:utf-8
import sys
import xml.sax

class StackContentHandler(xml.sax.ContentHandler):

    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
 
    def startElement(self, name, attrs):

        # only 'row' elements are relevant, skip elements that are not rows
        if name != "row":
            return
        
        id = "__none__"
        if attrs.has_key("Id"):
            id = attrs.getValue("Id")

        postId = "__none__"
        if attrs.has_key("PostId"):
            postId = attrs.getValue("PostId")

        score = "0"
        if attrs.has_key("Score"):
            score = attrs.getValue("Score")

        creationDate = "__none__"
        if attrs.has_key("CreationDate"):
            creationDate = attrs.getValue("CreationDate")

        userId = "__none__"
        if attrs.has_key("UserId"):
            userId = attrs.getValue("UserId")

        try:
            l = []
            l.append(id)
            l.append(postId)
            l.append(score)
            l.append(creationDate)
            l.append(userId)

            outfile.write('%s\n' % '|'.join(l))
        except:
            pass
 
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: " + sys.argv[0] + " <file>"
        sys.exit(1)

    fname = sys.argv[1]
    #print fname

    f = open(fname)
    outfile = open('CommentDate.txt','wb')
    xml.sax.parse(f, StackContentHandler())
    outfile.close()
    f.close()

