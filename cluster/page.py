from lxml import etree
import re

class Page:
    def __init__(self,path):
        self.path = path
        self.contents = open(path,"r").read().replace("\n","")
        self.xpaths = {}
        self.getXpaths()
        self.tfidf = {}

    def removeIndex(self,xpath):
    	indexes = re.findall(r"\[\d+\]",str(xpath))
    	for index in indexes:
			xpath = xpath.replace(index,"")
    	return xpath

    def addXpath(self,xpath):
    	if xpath in self.xpaths.keys():
    		self.xpaths[xpath] += 1
    	else:
    		self.xpaths[xpath] = 1

    def getXpaths(self):
    	tree= etree.HTML(str(self.contents))
    	Etree = etree.ElementTree(tree)
    	nodes = tree.xpath("//*[not(*)]")
    	for node in nodes:
    		# we do not consider index or predicate here
    		xpath = Etree.getpath(node)
    		xpath = self.removeIndex(xpath)
    		self.addXpath(xpath)

    def outputXpaths(self):
    	for key in self.xpaths.keys():
    		print key+"\t" + str(self.xpaths[key])

    def outputtfidf(self):
        for key in self.tfidf.keys():
            print "tf-idf of\t" + key+"\t" + str(self.tfidf[key])

    # add global xpaths
    def expandXpaths(self, global_xpaths):
    	for global_xpath in global_xpaths:
    		if global_xpath not in self.xpaths:
    			self.xpaths[global_xpath] = 0
    
    def updatetfidf(self,idf):
        # idf is a dict and given by allPages object
        for xpath in self.xpaths.keys():
            self.tfidf[xpath] = float(self.xpaths[xpath])*float(idf[xpath])



if __name__=='__main__':
	page_test = Page("../Crawler/toy_data/questions/question1.html")
	page_test2 = Page("../Crawler/toy_data/users/user1.html")
	for key in page_test.xpaths.keys():
		if key not in page_test2.xpaths.keys():
			print key
    

