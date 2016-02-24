from lxml import etree
import re
import  gensim.models
import copy
import math

class Page:
    def __init__(self,path):
        self.path = path.replace("_","/").replace(".html","")
        self.original = open(path,"r").read()
        self.contents = self.original.replace("\n","")
        self.xpaths = {} # tf
        self.xpaths_list = []
        self.dfs_xpaths_list = []
        self.filtered_dfs_xpaths_list = []
        self.getXpaths()
       # self.generalize_xpath()
        self.onehot = {}
        self.normonehot = {}
        self.tfidf = {}
        self.selected_tfidf = {}
        self.selected_logtfidf = {}
        self.normtfidf = {}
        self.logtfidf = {}
        self.embedding = []
        #self.getEmbedding()
        self.Leung = {}


    def generalize_xpath(self):
        # before expand xpath 
        self.generalize_paths = []
        temp = self.xpaths_list
        for i in range(len(temp)):
            for j in range(i+1,len(temp)):
                path = self.generalize(temp[i],temp[j])
                if path and path not in self.generalize_paths:
                    self.generalize_paths.append(path)

    def generalize(self,path1,path2):
        nodes_1 = path1.split("/")
        nodes_2 = path2.split("/")
        if len(nodes_1) != len(nodes_2):
            return None
        else:
            flag = 1
            pos = -1 # position of the unique diffrent nodes
            length = len(nodes_1)
            for i in range(length):
                if nodes_1[i] != nodes_2[i]:
                    if flag == 0:
                        return None
                    else:
                        flag -= 1
                        if self.removeIndex(nodes_1[i]) == self.removeIndex(nodes_2[i]):
                            pos = i
                        else:
                            return None
        if pos!= -1:
            for j in range(length):
                print nodes_1[j],
            for j in range(length):
                print nodes_2[j],

            print "\n"


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
            self.xpaths_list.append(xpath)
    		

    def getXpaths(self,index=False):
    	tree= etree.HTML(str(self.contents))
    	Etree = etree.ElementTree(tree)
    	nodes = tree.xpath("//*[not(*)]")
    	for node in nodes:
    		# we do not consider index or predicate here
            xpath = Etree.getpath(node)
            #self.dfs_xpaths_list.append(xpath) # except for this one
            if not index:
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
            self.logtfidf[xpath] = math.log(float(self.xpaths[xpath]+1),2)*float(idf[xpath])
            self.tfidf[xpath] = float(self.xpaths[xpath])*float(idf[xpath])
            if self.xpaths[xpath] > 0:
                self.onehot[xpath] = 1*float(idf[xpath])
            else:
                self.onehot[xpath] = 0
        # for visualization
        tfidf_sum = sum(self.tfidf.values())
        for item in self.tfidf:
            self.normtfidf[item] = float(self.tfidf[item])/tfidf_sum
        '''
        onehot_sum = sum(self.onehot.values())
        
        logtfidf_sum = sum(self.logtfidf.values())
        
            self.normlogtfidf[item] = float(self.logtfidf[item])/logtfidf_sum
            self.normonehot[item] = float(self.onehot[item])/onehot_sum
        '''
        # test
            
    def getAnchor(self):
        print "start getAnchor"
        link_dict = {}
        tree= etree.HTML(str(self.contents))
        Etree = etree.ElementTree(tree)
        nodes = tree.xpath("//a")
        for node in nodes:
            try:
                xpath = self.removeIndex(Etree.getpath(node))
                #print xpath,node.attrib['href']
                if xpath not in link_dict:
                    link_dict[xpath] = []
                else:
                    link_dict[xpath].append(node.attrib['href'])
            except:
                err = "Oh no! " + str(node)
        return link_dict

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

    def update_Leung(self,key):
        if self.xpaths[key] >0:
            self.Leung[key] = 1.0
        else:
            self.Leung[key] = 0.0

    def update_selected_tfidf(self,key):
        self.selected_tfidf[key] = copy.copy(self.tfidf[key])
        self.selected_logtfidf[key] = copy.copy(self.logtfidf[key])



if __name__=='__main__':
    #re_href = re.compile(r'(?<='href': ').*(?=')')
    print "Main for page.py"
    page_test = Page("../Crawler/toy_data/questions/question1.html")
    page_test2 = Page("../Crawler/toy_data/users/user1.html")
    page_test.getAnchor()
    

