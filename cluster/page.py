from lxml import etree
import re
import  gensim.models


class Page:
    def __init__(self,path):
        self.path = path.replace("_","/").replace("../Crawler/crawl/data/Questions/","").replace(".html","")
        self.original = open(path,"r").read()
        self.contents = self.original.replace("\n","")
        self.xpaths = {}
        self.dfs_xpaths_list = []
        self.getXpaths()
        self.onehot = {}
        self.normonehot = {}
        self.tfidf = {}
        self.normtfidf = {}
        self.embedding = []
        self.getEmbedding()

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
            #self.dfs_xpaths_list.append(xpath) # except for this one
            xpath = self.removeIndex(xpath)
            self.dfs_xpaths_list.append(xpath)
            self.addXpath(xpath)

    def outputXpaths(self):
        # by dfs order?
    	for key in self.xpaths.keys():
            if self.xpaths[key] >0:
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
            if self.xpaths[xpath] > 0:
                self.onehot[xpath] = 1*float(idf[xpath])
            else:
                self.onehot[xpath] = 0
        onehot_sum = sum(self.onehot.values())
        tfidf_sum = sum(self.tfidf.values())
        for item in self.tfidf:
            self.normtfidf[item] = float(self.tfidf[item])/tfidf_sum
            self.normonehot[item] = float(self.onehot[item])/onehot_sum
        # test
            
    def getAnchor(self):
        print "start getAnchor"
        tree= etree.HTML(str(self.contents))
        Etree = etree.ElementTree(tree)
        nodes = tree.xpath("//a")
        for node in nodes:
            try:
                print node.attrib['href']
            except:
                print "Oh no! " + str(node)

    def getEmbedding(self):
        # this should just load once , need to fix this later!
        model = gensim.models.Word2Vec.load("./Data/word2Vec.model")
        total = [0.0 for i in range(100)]
        n = 0
        for xpath in self.dfs_xpaths_list:
            try:
                total += model[xpath]
                n = n + 1
            except:
                error = "less than 5 times"
        avg_embedding = []
        for i in range(len(total)):
            avg_embedding.append(total[i]/float(n))
        # normalize
        self.embedding = avg_embedding



if __name__=='__main__':
    #re_href = re.compile(r'(?<='href': ').*(?=')')
    print "Main for page.py"
    page_test = Page("../Crawler/toy_data/questions/question1.html")
    page_test2 = Page("../Crawler/toy_data/users/user1.html")
    page_test.getAnchor()
    

