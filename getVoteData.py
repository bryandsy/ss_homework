# coding:utf-8
import sys
import xml.sax

class StackContentHandler(xml.sax.ContentHandler):

    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
 
    def startElement(self, name, attrs):
        positive = ['1', '2', '5', '7', '8', '11']
        negative = ['3', '4', '6', '9', '10', '12', '13']

        # only 'row' elements are relevant, skip elements that are not rows
        if name != "row":
            return

        vtype = "__none__"
        if attrs.has_key("VoteTypeId"):
            vtype = attrs.getValue("VoteTypeId")

        try:
            status = 1 if vtype in positive else 0
            outfile.write('%d|%s\n' % (status, attrs.getValue('CreationDate').split('T')[0] ))
        except:
            pass
 
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: " + sys.argv[0] + " <file>"
        sys.exit(1)

    fname = sys.argv[1]
    #print fname

    f = open(fname)
    outfile = open('VoteDate.txt','wb')
    xml.sax.parse(f, StackContentHandler())
    outfile.close()
    f.close()

