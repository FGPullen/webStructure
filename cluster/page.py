import copy
import math
import re

from lxml import etree


class Page:
    def __init__(self,path,mode="raw"):

        self.path = path.replace("_","/").replace(".html","")
        self.xpaths = {} # tf
        self.xpaths_list = []
        self.dfs_xpaths_list = []
        self.filtered_dfs_xpaths_list = []
        #self.getDFSXpaths(root)
        # self.generalize_xpath()
        self.onehot = {}
        self.normonehot = {}
        self.tfidf = {}
        self.selected_tfidf = {}
        self.selected_logtfidf = {}
        self.bigram_dict = {}
        self.normtfidf = {}
        self.logtfidf = {}
        self.embedding = []
        #self.getEmbedding()
        self.Leung = {}
        if mode == "raw":
        # getting xpaths from original content
            self.original = open(path,"r").read()
            self.contents = self.original.replace("\n","")
            self.getXpaths()
            root = etree.HTML(str(self.contents))
        #else:
         #   a = "read_features"
            #self.read_features()
    def read_tf_idf(self,features_str): #features_str is a string like num1 num2 num3 ...(space split)
        features = features_str.strip().split()
        for index,feature in enumerate(features):
            self.selected_tfidf[index] = float(feature)


    def read_log_tf_idf(self,features_str): #features_str is a string like num1 num2 num3 ...(space split)
        features = features_str.strip().split()
        for index,feature in enumerate(features):
            self.selected_logtfidf[index] = float(feature)


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

    def stemming(self,xpath):
        # heuristic xpath stemming
        tags = xpath.split('/')
        #for i in reversed(range(len(tags))):
        for i in reversed(range(len(tags))):
            if tags[i] == "p":
                #print xpath
                #print '/'.join(tags[:i+1])
                return '/'.join(tags[:i+1])
            if tags[i] == "ul" or tags[i] == "ol":
                #print xpath
                #print '/'.join(tags[:i+1])
                return '/'.join(tags[:i+1])
            if tags[i] == "table":
                return '/'.join(tags[:i+1])

        return xpath

    def getXpaths(self,index=False):
        # TODO: XPaths are pretty deep and it becomes noisier
        # when it goes deeper.  Pruning might be a good idea.
        tree= etree.HTML(str(self.contents))
        Etree = etree.ElementTree(tree)
        nodes = tree.xpath("//*[not(*)]")
        for node in nodes:
            # we do not consider index or predicate here
            xpath = Etree.getpath(node)
            #self.dfs_xpaths_list.append(xpath) # except for this one
            if not index:
                xpath = self.removeIndex(xpath)
            #xpath = "/".join(xpath.split('/')[:-1]) # prune the leaf level

            #xpath = self.stemming(xpath)
            #if s_xpath != xpath:
            #    self.dfs_xpaths_list.append(s_xpath)
            #    self.addXpath(s_xpath)

            #print xpath
            self.dfs_xpaths_list.append(xpath)
            self.addXpath(xpath)

    def detect_loop(self, root):
        return False
        if len(root) < 2:
            return False
        prev = None
        for child in root:
            if prev is not None:
                if child.tag != prev.tag:
                    return False
                if child.get('class') != prev.get('class'):
                    return False
            prev = child
        return True

    def getDFSXpaths(self, root, xpath=""):
        loop_node = self.detect_loop(root)
        for node in root:
            if type(node.tag) is not str:
                continue
            if loop_node:
                new_xpath = "/".join([xpath, node.tag, 'loop'])
            else:
                #print node.get('class')
                tag = node.tag+"[" + str(node.get('class')) + "]"
                new_xpath = "/".join([xpath, tag])
            if len(node) == 0:
                #print new_xpath
                self.dfs_xpaths_list.append(new_xpath)
                self.addXpath(new_xpath)
            if len(node) != 0:
                self.getDFSXpaths(node, new_xpath)
            if loop_node:
                break

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

    def get_bigram_features(self,bigram_list):

        for bigram in bigram_list:
            path1, path2 = bigram[0],bigram[1]
            if self.xpaths[path1] >0 and self.xpaths[path2] >0:
                self.bigram_dict["("+path1+","+path2+")"] = 1
            else:
                self.bigram_dict["("+path1+","+path2+")"] = 0


    def getAnchor(self,use_attrib=False):
        print "start getAnchor"
        link_dict = {}
        tree= etree.HTML(str(self.contents))
        Etree = etree.ElementTree(tree)
        nodes = tree.xpath("//a")
        for node in nodes:
            #print type(node)
            if 'class' in node.attrib:
                attrib = node.attrib['class']
                #print node.attrib
            else:
                attrib = ""
            try:
                xpath = self.removeIndex(Etree.getpath(node))
                if use_attrib:
                    xpath += "[{}]".format(attrib)
                #print xpath,node.attrib['href']
                if xpath not in link_dict:
                    link_dict[xpath] = []
                link_dict[xpath].append(node.attrib['href'])
            except:
                err = "Oh no! " + str(node)
        #print link_dict
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
    #page_test = Page("./Data/test.html")
    page_test = Page("../Crawler/Mar15_samples/stackexchange/http:__android.stackexchange.com_users_11343_mr-buster.html")
    page_test.getAnchor()
