import os.path
import sys
import scipy.sparse as sps
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from hits_estimate import read_trans_dict
from pageCluster import pageCluster
from page import Page
from auto_crawler import crawler
import math,pickle
from collections import defaultdict,Counter
import re
from sklearn.preprocessing import normalize
import numpy as np
import lxml
import traceback
from bisect import bisect
from url_annotator import annotator
import random,time
from sample import sampler

class vidal:


    def __init__(self, dataset, date, entry,prefix, cluster_rank, crawl_size):
        self.dataset = dataset
        self.date = date
        self.cluster_rank = cluster_rank
        self.crawl_size = crawl_size
        self.cluster_rank = cluster_rank
        self.entry, self.prefix = entry,prefix
        self.history_set = set()
        self.group_list = []
        self.group_dict = {}
        if self.date == "May1":
            self.path_prefix = "../../Crawler/{}_samples/{}/".format(date,dataset.replace("new_",""))
        else:
            self.path_prefix = "../Crawler/{}_samples/{}/".format(date,dataset)
        self.folder_path = ["../../Crawler/{}_samples/{}/".format(date,dataset.replace("new_",""))]
        self.sitemap = pageCluster(dataset,date,self.folder_path,0)
        self.cluster_num = int(self.sitemap.DBSCAN())
        self.full_folder = "../../Crawler/full_data/" + dataset
        c = crawler(self.dataset,self.date,None,None,eps=None,cluster_rank=self.cluster_rank,crawl_size=None,rank_algo=None)
        self.target_cluster = c.target_cluster
        self.crawler = c

    def get_bfs_file(self):
        file_path = "./results/vidal/{0}_{1}_{2}_vidal_size1001.txt".format(self.dataset,self.date,self.cluster_rank)
        print "bfs file is ", file_path
        self.leaf_list = []
        lines = open(file_path,"r").readlines()
        print len(lines)," reading bfs files"
        self.url_list, self.path_list = [], []
        # diction : key:url value: id
        self.diction, self.parent_dict = {},{}
        self.train_size = 1000
        for i in range(self.train_size):
            line = lines[i].strip()
            tmp = line.split("\t")
            url, parent_url, cluster_id = tmp[0], tmp[1], int(tmp[-1])
            print tmp
            # this page should not be considered since its parent is a leaf node

            if i == 0:
                self.url_list.append(url)
                self.diction[url] = i
                self.parent_dict[url] = -1
                continue

            self.url_list.append(url)
            self.diction[url] = i
            self.parent_dict[i] = self.diction[parent_url]

            if cluster_id == self.target_cluster:
                print cluster_id, self.target_cluster, "!!!!!"
                path= self.get_path(i)
                self.output(path)
                self.path_list.append(path)
                self.leaf_list.append(i)



    def output(self, path_list):
        for id in path_list:
            try:
                print self.url_list[id],
            except:
                print "wrong is {}",id
        print ""
        for id in path_list:
            print id,
        print ""

    def get_path(self,id):
        path_list = [id]
        while(id!=0):
            id = self.parent_dict[id]
            path_list.append(id)
        return path_list


    def get_sample_cluster(self):
        # The below is for the first version
        if self.dataset=="youtube":
            cluster_list = ["../../Crawler/May1_samples/youtube/https:__www.youtube.com_watch?v=zhnMSVb0oYA.html",\
                            "../../Crawler/May1_samples/youtube/https:__www.youtube.com_user_vat19com_discussion.html",\
                            "../../Crawler/May1_samples/youtube/https:__www.youtube.com_playlist?list=PLuKg-WhduhkmIcFMN7wxfVWYu8qnk0jMN.html"
                            ]
            file_path = cluster_list[self.cluster_rank]
        elif self.dataset == "stackexchange":
            cluster_list =["../../Crawler/May1_samples/stackexchange/http:__android.stackexchange.com_q_138084.html",\
            "../../Crawler/May1_samples/stackexchange/http:__android.stackexchange.com_users_134699_rood.html",\
            "../../Crawler/May1_samples/stackexchange/http:__android.stackexchange.com_search?q=user:1465+[2.2-froyo].html"]
            file_path = cluster_list[self.cluster_rank]
        elif self.dataset == "asp":
            cluster_list =["../../Crawler/May1_samples/asp/http:__forums.asp.net__post_6027404.aspx.html",\
            "../../Crawler/May1_samples/asp/http:__forums.asp.net__members_sapator.aspx.html",\
                           "../../Crawler/May1_samples/asp/http:__forums.asp.net__login_RedirectToLogin?ReturnUrl=_post_set_15_2087379_6028949.html"]
            file_path = cluster_list[self.cluster_rank]
        elif self.dataset == "tripadvisor":
            cluster_list = ["../../Crawler/May1_samples/tripadvisor/http:__www.tripadvisor.com_Hotel_Review-g48016-d655840-Reviews-Park_Lane_Motel-Lake_George_New_York.html.html",\
                            "../../Crawler/May1/samples/tripadvisor/http://www.tripadvisor.com/ShowUserReviews-g32578-d77436-r354322139-The/Lodge/at/Torrey/Pines-La/Jolla/San/Diego/California#review/354322139"]
            file_path = cluster_list[self.cluster_rank]
        elif self.dataset == "douban":
            cluster_list = ["../../Crawler/July30_samples/douban/https:__movie.douban.com_review_7363997_.html",\
                            "../../Crawler/July30_samples/douban/https:__movie.douban.com_celebrity_1303681_.html"]
            file_path = cluster_list[self.cluster_rank]
        elif self.dataset == "hupu":
            cluster_list = ["../../Crawler/July30_samples/hupu/http:__voice.hupu.com_nba_1482921.html.html",\
                            "../../Crawler/July30_samples/hupu/http:__voice.hupu.com_people_4915254.html"]
            file_path = cluster_list[self.cluster_rank]
        elif self.dataset == "rottentomatoes":
            cluster_list = ["../../Crawler/July30_samples/rottentomatoes/http:__www.rottentomatoes.com_m_macbeth_2015_.html","../../Crawler/July30_samples/rottentomatoes/http:__www.rottentomatoes.com_celebrity_peter_nowalk_.html"]
            file_path = cluster_list[self.cluster_rank]

        #elif self.dataset == "":
        #    cluster_list = []

    # @ input: page object and self.sitemap
    # @ return: cluster id
    def classify(self,file_path):
        #self.sitemap
        page = Page(file_path)
        x = []
        for feat in self.sitemap.features:
            if feat in page.xpaths:
                # remember log !
                x.append(math.log(float(page.xpaths[feat]+1),2) * self.sitemap.UP_pages.idf[feat])
            else:
                x.append(0.0)
        pred_y = self.sitemap.nbrs.predict(x)[0]
        return page,pred_y

    def calculate_url_similarity(self,id1,id2):
        u1, u2 = self.url_list[id1], self.url_list[id2]
        u1 = u1.replace("//","/")
        u2 = u2.replace("//","/")
        list_1 = u1.split("/")
        list_2 = u2.split("/")
        #print list_1
        #print list_2
        if len(list_1) != len(list_2):
            return False
        if list_1[1] != list_2[1] or list_1[0] != list_2[0]:
            return False
        diff_num = 0
        for i in range(len(list_1)):
            if list_1[i] != list_2[i]:
                diff_num += 1
        if diff_num > 2:
            return False
        return True

    # find the children of each node and save them  to diction
    def get_node_dict(self):
        self.node_dict = defaultdict(set)
        for path in self.path_list:
            for i in range(len(path)-1,0,-1):
                self.node_dict[path[i]].add(path[i-1])
        print "output parent-child pair"
        for key in self.node_dict:
            print key, self.node_dict[key]


    # input: self.path_list
    def group(self,list_of_nodes=[0]):
        children_list = []
        for node in list_of_nodes:
            node_set = self.node_dict[node]
            for child in node_set:
                if child not in children_list:
                    children_list.append(child)
        print children_list
        return children_list

    def segment(self,node_list):
        list = []
        #print len(node_list)
        for i in range(len(node_list)):
            id = node_list[i]
            if i == 0:
                list.append([id])
            else:
                flag = False
                for index,set in enumerate(list):
                    tmp = self.judge(id,set)
                    if tmp:
                        set.append(id)
                        flag = True
                        break
                if flag is False:
                    list.append([id])
            #print i, list
        return list

    def judge(self,id,set):
        flag = True
        for key in set:
            if not self.calculate_url_similarity(id,key):
                flag = False
        return flag

    # iteratively find the
    def run(self,start=[0],parent_id=-1):
        self.group_list.append(start)
        current_id = len(self.group_list)-1
        self.group_dict[current_id] = parent_id
        children_list = self.group(start)
        groups = v.segment(children_list)
        for group in groups:
            self.run(group,current_id)


    def get_best_path(self):
        max_leaf, max_id = -1, -1
        for g_id, group in enumerate(self.group_list):
            num = 0
            for id in group:
                if self.leaf_judge(id):
                    num +=1
            if num > max_leaf:
                max_id = g_id
                max_leaf = num
            print g_id, num
        return max_id

    def leaf_judge(self,pid):
        if pid in self.leaf_list:
            return True
        else:
            return False


    def get_pattern(self,dataset,cid):
        if self.dataset == "stackexchange":
            if cid == 0:
                self.pattern = [["questions"],["^questions?(.*)=(.*)$"],["questions","^[0-9]+$","^(.+)$"]]
            elif cid == 1:
                self.pattern = [["questions"],["questions","^[0-9]+$","^(.+)$"],["q","^[0-9]+$"],["users","^[0-9]+$","^(.+)$"]]
        if self.dataset == "asp":
            if cid == 0:
                #self.pattern = [[""],["^default.aspx$","^[0-9]+?(.*)$"],["t","^[0-9]+.aspx(.+)$"]]
                self.pattern = [[""],["^members$","^(.*).aspx$"],["p","^[0-9]+$","^[0-9]+.aspx(.*)$"]]
            elif cid == 1:
                #self.pattern = [[""],["^members$","^(.*).aspx$"],["t","^[0-9]+.aspx(.+)$"]]
                self.pattern = [[""],["^default.aspx$","^[0-9]+?(.*)$"],["^members$","^(.*).aspx$"]]
        if self.dataset == "youtube":
            if cid == 0:
                self.pattern = [[""],["^user|channel$","^(.*)$"],["^watch\?v=(.*)$"]]
            elif cid ==  1:
                self.pattern = [[""],["^user$","^(.*)$"],["^channel$","^(.*)$"]]
        if self.dataset == "hupu":
            if cid == 0:
                self.pattern =  [["hot"],["^(.*)$"],["^(.*)$","^[0-9]+.html(.*)$"]]
            elif cid == 1:
                self.pattern = [["hot"],["^(.*)$","^(.*)$"]]
        if self.dataset == "douban":
            if cid == 0:
                self.pattern = [[""],["subject","^[0-9]+$","^\?from=(.*)$"],["review","^[0-9]+$"]]
            elif cid == 1:
                self.pattern = [[""],["subject","^[0-9]+$"]]
        if self.dataset == "rottentomatoes":
            if cid == 0:
                self.pattern = [[""],["^(.*)$"],["m","^(.*)$"]]
            elif cid == 1:
                self.pattern = [[""],["^(.*)$"],["m","^(.*)$"],["celebrity","^(.*)$"]]



    def crawl(self):
        self.get_pattern(self.dataset,self.cluster_rank)
        self.a = annotator(self.dataset)

        write_file = open("./results/vidal_{0}_{1}_{2}_size{3}.txt".format(self.dataset,self.date, self.cluster_rank, self.crawl_size),"w")
        num_web_crawl = 0
        entry,prefix = self.entry, self.prefix
        self.url_stack  = [(entry,"",0)]
        self.final_list = []
        size, num = self.crawl_size,  0 # the number of crawling
        s = sampler(self.dataset,self.entry,self.prefix,0)
        while(num<size and len(self.url_stack)>0):
            first_url = self.url_stack[0][0]
            parent_url = self.url_stack[0][1]
            rule_id = self.url_stack[0][2]
            try:
                print "first url is ",first_url
            except:
                traceback.print_exc()

            if first_url not in self.history_set:
                num += 1
                try:
                    url_list, new_rule_id = self.crawl_link(first_url, rule_id,self.history_set, s)
                    self.final_list.append((first_url,parent_url,rule_id))
                except:
                    print "might miss somthing here"
                    traceback.print_exc()
                    flag = s.crawlUrl(first_url,self.dataset,self.url_stack,self.history_set)
                    if flag == 1:
                        url_list, new_rule_id = self.crawl_link(first_url, rule_id,  self.history_set,s )
                        self.final_list.append((first_url,parent_url,rule_id))
                        random_time_s = random.randint(5, 10)
                        time.sleep(random_time_s)
                        num_web_crawl += 1
                        if num_web_crawl%10 == 9:
                            random_time_s = random.randint(60, 90)
                            time.sleep(random_time_s)
                    else:
                        num -= 1
                        print "crawl failure"
            if self.url_stack[0][0] == first_url:
                self.url_stack.pop(0)
            print " num is {}".format(num)
            sys.stdout.flush()
            self.history_set.add(first_url)

        print len(self.final_list), "length of final list"

        for pair in self.final_list:
            url, parent_url,cluster_id = pair[0],pair[1],pair[2]
            write_file.write(url + "\t" + str(parent_url) + "\t"+ str(cluster_id) + '\n')



    def crawl_link(self, first_url, rule_id, history_stack, sampler):
        #rule_id = self.get_rule_id(first_url)
        print first_url, "in town!"
        # already the leaf node, no need to crawl
        if rule_id != len(self.pattern)-1:
            rule_id += 1

        file_path = self.full_folder + "/" + first_url.replace("/","_") +".html"
        #print file_path
        page,cluster_id = self.crawler.classify(file_path)
        available_url_list = []
        print file_path, cluster_id , "file_path for cluster_id"
        original = open(file_path,"r").read()
        contents = original.replace("\n","")
        link_dict = sampler.getAnchor(contents,first_url,sample_flag=False)

        for xpath in link_dict:
            link_list = link_dict[xpath]
            for url in link_list:
                if self.judge_match(url,rule_id):
                    if url not in history_stack and url not in available_url_list:
                        available_url_list.append((url,first_url,rule_id))
                        print url,rule_id,self.pattern[rule_id]
        print available_url_list
        self.url_stack += available_url_list
        return available_url_list, rule_id


    #def get_rule_id(self,first_url):
    #    self.pattern


    def judge_match(self,url,rule_id):
        rule = self.pattern[rule_id]
        url = url.replace("//","/").strip()
        if url[-1] == "/":
            url = url[:-1]
        print url, rule, " url and rule"
        if len(rule) != (len(url.split("/"))-2):
            print "length doest not match", len(rule), len(url.split("/"))-2
            return False
        else:
            flag = self.a.match(url,rule)
            print "flag is ", flag
            return flag




if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",help="The dataset to crawl")
    parser.add_argument("date",help="The date of sampling")
    parser.add_argument('entry', help='The entry page')
    parser.add_argument('prefix', help='For urls only have partial path')
    parser.add_argument('cluster_rank',help="The cluster number for crawling")
    parser.add_argument('crawl_size',help="crawling size")
    args = parser.parse_args()

    v = vidal(args.dataset, args.date, args.entry, args.prefix, int(args.cluster_rank), int(args.crawl_size))
    print v.target_cluster, " target cluster is "

    if v.crawl_size == 0:

        v.get_bfs_file()
        print v.path_list, "path list for v"


        v.get_node_dict()
        v.run()
        print "look into the entry's children"

        for key in v.group_dict:
            if key == 0:
                l = v.group_dict[key]
                print v.group_list[l]

        print v.group_list
        print v.group_dict


        max_id = v.get_best_path()
        print "leaf list", len(v.leaf_list), v.leaf_list
        for id in v.leaf_list:
            print id, v.url_list[id]
        # the group that has the largest number of leaf node
        print max_id, v.group_list[max_id] , " Max!"
        id = max_id
        while(id !=-1):
            print id, v.group_list[id],
            for i in v.group_list[id]:
                print v.url_list[i],
            print ""
            id = v.group_dict[id]
        '''
        v.get_node_dict()
        list = v.group()
        group =  v.segment(list)

        for tmp in group:
            print tmp
            for id in tmp:
                print v.url_list[id],
            print ""
        '''
    else:
        v.crawl()
