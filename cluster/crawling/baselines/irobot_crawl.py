import os.path
import sys
import scipy.sparse as sps
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import math,pickle
from collections import defaultdict,Counter
import re
from sklearn.preprocessing import normalize
import numpy as np
from lxml import etree
import traceback
from bisect import bisect
import urllib2
import random,time
from sample import sampler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import LSHForest

text_threshold = 0.8

class irobot:
    def __init__(self):
        self.crawl_num = 1000
        self.history_set = set()

    def update(self,dataset):
        self.dataset = dataset
        self.folder_path = "../../../Crawler/July30_samples/{}".format(self.dataset)
        # initiate self.category and self.ground_truth and self.path_list
        self.input_file = "../July30/site.dbscan/{}.txt".format(self.dataset)
        #self.input_file = "../../baselines/data/Clustering/{}.txt".format(self.dataset)
        self.cluster_dict = self.initialization(self.input_file)
        self.full_folder = "../../../Crawler/full_data/" + dataset
        self.get_entry_prefix()
        # give weights to each cluster by :(1) ratio (2) average size (3) duplication ratio
        # output:
        #self.weight_clusters()
        #
        #self.generate_traversal_path()

    def get_entry_prefix(self):
        if self.dataset == "stackexchange":
            self.entry, self.prefix =  "http://android.stackexchange.com/questions", "http://android.stackexchange.com"
        elif self.dataset == "asp":
            self.entry, self.prefix = "http://forums.asp.net/","http://forums.asp.net"
        elif self.dataset == "youtube":
            self.entry, self.prefix =  "https://www.youtube.com/","https://www.youtube.com"
        elif self.dataset == "hupu":
            self.entry, self.prefix = "http://voice.hupu.com/hot","http://voice.hupu.com"
        elif self.dataset == "rottentomatoes":
            self.entry, self.prefix = "http://www.rottentomatoes.com","http://www.rottentomatoes.com"
        elif self.dataset == "douban":
            self.entry, self.prefix = "http://movie.douban.com","http://movie.douban.com"


    def initialization(self,input_file):
        cluster_dict = defaultdict(list)
        prefix = "../../Crawler/July30/samples/{}/".format(self.dataset)
        lines = open(input_file,"r").readlines()
        for line in lines:
            line = line.replace(prefix,"")
            temp = line.strip().split()
            url, cluster_id = temp[0], temp[-1].replace("cluster:","")
            cluster_dict[cluster_id].append(url)
        print cluster_dict
        return cluster_dict


    # we use the method of double-end crawling for initialling sampling training data
    #def initial_sampling(self):
    # input: entry, prefix, site, trans_xpath_dict , target_cluster id
    # target: find the cluster page that we want
    def crawling(self,crawl_size=1000):
        if not os.path.exists("./results/irobot/"):
            os.mkdir("./results/irobot/")
        write_file = open("./results/irobot/{0}_irobot_size{1}.txt".format(self.dataset,crawl_size),"w")
        entry,prefix = self.entry, self.prefix
        self.url_stack  = [(entry,"",0)] #(entry,parent_url,crawl_level)
        self.final_list = []
        size, num = crawl_size, 0 # the number of crawling
        crawl_id = 0
        s = sampler(self.dataset,self.entry,self.prefix,0)
        end = 0
        num_web_crawl = 0
        while(num<size and len(self.url_stack)>0):
            print self.url_stack[-1]
            print self.url_stack[0]
            first_url = self.url_stack[end][0]
            parent_url = self.url_stack[end][1]
            crawl_level = self.url_stack[end][2]

            try:
                print "first url is ",first_url

            except:
                traceback.print_exc()

            if first_url not in self.history_set:
                num += 1
                try:
                    url_list = self.crawl_link(first_url, crawl_level, self.history_set, s)
                    print "url list", len(url_list)
                    self.url_stack.pop(end)
                    self.url_stack += url_list
                    self.final_list.append((first_url,parent_url,crawl_level))
                except:
                    print "might miss somthing here"
                    traceback.print_exc()
                    flag = self.crawlUrl(first_url,self.dataset,self.url_stack,self.history_set)
                    if flag == 1:
                        url_list = self.crawl_link(first_url, crawl_level,  self.history_set,s )
                        self.url_stack.pop(end)
                        print "url list", len(url_list)
                        self.url_stack += url_list
                        self.final_list.append((first_url,parent_url,crawl_level))
                        random_time_s = random.randint(5, 10)
                        time.sleep(random_time_s)
                        num_web_crawl += 1
                        if num_web_crawl%10 == 9:
                            random_time_s = random.randint(60, 90)
                            time.sleep(random_time_s)
                    else:
                        num -= 1
                        print "crawl failure"
            else:
                self.url_stack.pop(end)

            end = random.choice([0,-1])
            print "end is ", end
            crawl_id += 1
            print " num is {}".format(num)
            sys.stdout.flush()
            if num >= size:
                print "crawl_id is {0} for size {1}".format(crawl_id,size)

            self.history_set.add(first_url)
        print len(self.final_list), "length of final list"

        for pair in self.final_list:
            url, parent_url, crawl_level = pair[0],pair[1],pair[2]
            write_file.write(url + "\t" + str(parent_url) +"\t" + str(crawl_level) + '\n')

    def crawl_link(self, first_url, crawl_level, history_stack, sampler):
        file_path = self.full_folder + "/" + first_url.replace("/","_") +".html"
        original = open(file_path,"r").read()
        contents = original.replace("\n","")
        tree= etree.HTML(str(contents))
        Etree = etree.ElementTree(tree)
        nodes = tree.xpath("//a")
        url_list = []
        for node in nodes:
            if "href" in node.attrib:
                url = node.attrib['href']
                if sampler.intraJudge(url,sampler.dataset):
                    url = sampler.transform(url)
                    if url not in history_stack and url not in url_list:
                        url_list.append((url,first_url,crawl_level+1))
        return url_list

    def crawlUrl(self, url, site, url_stack, history_stack):
        if url in history_stack:
            print "Already crawled!"
            return 0
        else:
            try:
                response = urllib2.urlopen(url, timeout=30)
                lines = response.read().replace("\n", "")
                #folder_path = "/bos/usr0/keyangx/webStructure/Crawler/full_data/" + site + "/"
                folder_path = "../../../Crawler/full_data/" + site + "/"
                file_name = folder_path + url.replace("/", "_") + ".html"
            except:
                traceback.print_exc()
                print " error in crawlUrl"
                return 0
        #if ".html.html" in file_name:
        #    file_name = file_name.replace(".html.html", ".html")
        print file_name
        if os.path.isfile(file_name):
            print "Already"
            return 0
        try:
            write_file = open(file_name, 'w')
            write_file.write(lines)
        except:
            return 0
        print "succesfully crawled missing page!"
        return 1

    # input: dict[key] -> list[path_list]
    def weight_clusters(self,cluster_dict):
        total_sum = 0
        for id,list in cluster_dict.iteritems():
            total_sum += len(list)
        ratio_weight = defaultdict(float)
        # the ratio / the size ratio of the cluster
        # cluster_id -> id
        for id,list in cluster_dict.iteritems():
            ratio_weight[id] = len(list)/float(total_sum)
        print ratio_weight

        # the average size of file / duplication number
        total_size = 0.0
        size_weight = defaultdict(float)
        duplication_weight = defaultdict(float)
        text_list = [] # on cluster level

        for id,list in cluster_dict.iteritems():
            print id, list

            for name in list:
                file_path = self.transform_to_path(name)
                #file_path = self.folder_path + "/" + name
                size,text = self.parse_file(file_path)
                text_list.append(text)
                size_weight[id] += size
                total_size += size
            #duplication_number = self.calculate_duplication_number(text_list)
            #duplication_weight[id] = 1 - float(duplication_number)/float(len(list))
            print id, duplication_weight[id]

        self.calculate_duplication_number(text_list)

        avg_size = total_size/float(total_sum)

        for id,total_size in size_weight.iteritems():
            avg = total_size/float(len(cluster_dict[id]))
            size_weight[id] = avg/avg_size # avg_size is global


        cluster_weight = defaultdict(float)
        for id in self.cluster_dict:
            cluster_weight[id] = size_weight[id] * ratio_weight[id] * duplication_weight[id]

        return cluster_weight

    def transform_to_path(self,name):
        path = self.folder_path + "/" + name.replace("/","_") + ".html"
        return path

    def parse_file(self,file_path):
        try:
            if self.dataset == "hupu":
                file_path = file_path.replace("#",".html#")
            file_size = os.path.getsize(file_path)
        except:
            #print traceback.print_exc()
            file_path = file_path +".html"
            file_size = os.path.getsize(file_path)
        with open(file_path) as fin:
            content = fin.read()
        #print file_path
        text = ""
        page_tree = etree.HTML(content)
        nodes = page_tree.xpath("//a/text()")
        text  = " ".join( [ node for node in nodes ] )
        #print "text is ", text
        return file_size, text

    #def calculate_duplication(self,text_list):

    def LCS(self,text1, text2):
        m = len(text1)
        n = len(text2)
        #print m,n
        if len(text1) < 0.99* len(text2) or len(text2) < len(text1) * 0.99:
            return 0
        else:
            return 1
        '''
        else:
            X, Y = text1, text2
            # An (m+1) times (n+1) matrix
            C = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if X[i-1] == Y[j-1]:
                        C[i][j] = C[i-1][j-1] + 1
                    else:
                        C[i][j] = max(C[i][j-1], C[i-1][j])
            return float(C[m][n])/min(m,n)
        '''


    def calculate_duplication_number(self,text_list):
        print "length is ", len(text_list)
        tf_vectorizer = CountVectorizer(stop_words=None,analyzer='word',ngram_range=(5,5))
        #print text_list
        tf = tf_vectorizer.fit_transform(text_list)
        #print tf_vectorizer.get_feature_names()
        print tf[0]
        #print tf[123]
        lshf = LSHForest()
        #print tf
        lshf.fit(tf)
        distance,index = lshf.kneighbors(tf,n_neighbors=1)
        print distance, index





    '''
    def calculate_duplication_number(self,text_list):
        length = len(text_list)
        print " length is ", length
        group_list = [0]
        id_sample_list = [text_list[0]]
        for i in range(1,length):
            text = text_list[i]
            flag = 0
            for id, sample in enumerate(text_list):

                score = self.LCS(sample,text)
                #if score > text_threshold:
                if score==1:
                    group_list.append(id)
                    flag = 1
                    break
            if flag == 0:
                group_list.append(len(id_sample_list)+1)
                id_sample_list.append(text)
        c = Counter(group_list)
        print c
        return c.most_common(1)[0][1]
        '''


if __name__ == "__main__":
    dataset = sys.argv[1]
    function = sys.argv[2]
    r = irobot()
    r.update(dataset)
    if function == "learning":
        r.crawling()
    else:
        weight_dict = r.weight_clusters(r.cluster_dict)
        weight_list = [0.0 for i in range(len(weight_dict)+1)]
        for key,value in weight_dict.iteritems():
            weight_list[int(key)] = float(value)
        weights = np.array(weight_list)
        print weights
        std = np.std(weights)
        avg = np.mean(weights)
        threshold = avg - std
        print "threshold ", threshold
        for id, value in enumerate(weights):
            if id == -1:
                continue
            if float(value) < threshold:
                print id, value


