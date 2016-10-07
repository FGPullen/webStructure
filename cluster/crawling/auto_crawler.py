import os.path
import sys
import scipy.sparse as sps
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from hits_estimate import read_trans_dict
from pageCluster import pageCluster
from page import Page
import math,pickle
from collections import defaultdict,Counter
import re
from sklearn.preprocessing import normalize
import numpy as np
import lxml
import traceback
from bisect import bisect
import random,time
from sample import sampler
import logging


class crawler:

    def __init__(self, dataset, date, entry,prefix, eps, cluster_rank,crawl_size, num_sample=None, rank_algo="bfs"):
        self.dataset = dataset
        self.date = date
        self.eps = eps
        self.cluster_rank = cluster_rank
        self.rank_algo = rank_algo
        self.crawl_size = crawl_size
        #self.rules = self.get_rules()
        self.entry, self.prefix = entry,prefix
        self.history_set = set()
        self.history_stack = []
        self.num_samples = num_sample
        #if self.date == "May1":
        if num_sample is None:
            self.path_prefix = "../../Crawler/{}_samples/{}/".format(date,dataset.replace("new_",""))
        else:
            self.path_prefix = "../../Crawler/{0}_samples/{1}/{2}/".format(date,num_sample,dataset)
        #else:
        #    self.path_prefix = "../Crawler/{}_samples/{}/".format(date,dataset)
        if num_sample is None:
            self.folder_path = "../../Crawler/{}_samples/{}/".format(date,dataset)
        else:
            self.folder_path = "../../Crawler/{0}_samples/{1}/{2}/".format(date,num_sample,dataset)

        self.sitemap = pageCluster(dataset,date,self.folder_path,num_samples=num_sample)
        self.full_folder = "../../Crawler/full_data/" + dataset
        self.trans = {}
        self.queue = {}
        self.crawled_cluster_count = defaultdict(int)
        self.trans_dict = read_trans_dict(dataset,date)

        #self.cluster_dict = get_cluster_dict(dataset,date)
        #self.cluster_num = int(self.sitemap.DBSCAN(eps_val=self.eps))
        self.cluster_num = int(self.sitemap.DBSCAN()) + 1
        self.build_gold_cluster_dict()
        self.cluster_xpath_trans = self.get_xpath_transition() # and calculate self.navigation_table
        self.transition_mat = self.get_transition_matrix() # transition probability
        self.adjacency_mat = self.get_adjacency_matrix() # get the average adjacency links/or total links
        #self.trans_prob_mat = self.calculate_trans_prob_mat()
        self.target_cluster = self.get_sample_cluster()
        self.trans_mat = self.trans_dict_to_matrix() # need normalization
        #self.trans_mat = self.transition_mat
        self.max_score = 500

        #self.target_cluster = 0
        #print self.cluster_xpath_trans, "self.cluster_xpath_trans"

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

    # @ input: sitemap and (page,xpath)
    # @ (cluster,xpath) - > which cluster?
    def get_xpath_transition(self):

        sampled_urls = self.gold_dict.keys()
        counts_dict = defaultdict(int)  # (cluster_id, xpath) -> int / or simply (cluster_id )?
        xpath_counts_dict = defaultdict(lambda : defaultdict(float)) # (cluster_id, xpath) - > dict[cluster_id] -> int

        trans_dict = read_trans_dict(self.dataset,self.date)  # [page][xpath] = [url list] ->[cluster][xpath] = {probability list}
        #trans_dict = self.get_trans_dict(self.dataset,self.date)
        self.debug_file = open("trans_dict.debug","w")


        #print "sample_url", sampled_urls
        #print trans_dict, "trans_dict"
        #print self.gold_dict, "gold dict"
        for page_path, trans in trans_dict.iteritems():

            self.debug_file.write(page_path + "\n")
            page_url = page_path.replace(".html","").replace("_","/")
            if page_url not in self.gold_dict:
                #print "?" + page_url, " is missing", self.gold_dict.keys()[0]
                continue
            else:
                #print page_path, 'page pat'
                cluster_id = self.cluster_dict[page_url]
                counts_dict[cluster_id] += 1

            for xpath,url_list in trans.iteritems():
                length = len(url_list)
                count = 0
                for path in url_list:
                    if type(path) is tuple:
                       path = path[0]
                    url = path.replace(".html","").replace("_","/")
                    if url in sampled_urls:
                        count += 1
                #counts_dict[cluster_id] += 1
                key = (cluster_id,xpath)
                #print "for xpath: {0} --- {1} out of {2} have been crawled and have cluster id".format(xpath,count, length)
                if count == 0:
                    #counts_dict[key] += 1
                    self.debug_file.write(xpath + "\t" + str(len(url_list)) + " with no ground truth file" + "\n")
                    continue
                else:
                    #if cluster_id == 1:
                    #    print page_path, xpath, url_list, "xpath_url_list in train"
                    #key = (cluster_id,xpath)
                    #if key == (1,"/html/body/div/div/div/div/div/div/div/div/div/div/ul/li/div/div/div/div/div/div/div/ul/li/div/div/div/h3/a[yt-uix-sessionlink yt-uix-tile-link  spf-link  yt-ui-ellipsis yt-ui-ellipsis-2]"):
                    #    print page_path,url_list, "why 9 not 7???"
                    #counts_dict[key] += 1

                    ratio = float(length)/float(count)
                    self.debug_file.write(xpath + "\t" + str(count) + "\t" + str(length)+ "\t" + str(url_list) + "\n")

                    for path in url_list:
                        if type(path) is tuple:
                            path = path[0]
                        #print path, "it is the url"
                        url = path.replace(".html","").replace("_","/")
                        if url in sampled_urls:
                            destination_id = self.cluster_dict[url]
                            #print url, destination_id
                            xpath_counts_dict[key][destination_id] += 1 * ratio
                    #if cluster_id == 1:
                    #    print ""

        # average
        #for key in xpath_counts_dict.keys():
        #    print xpath_counts_dict[key], "xpath counts dict"
        '''
        for key,count in counts_dict.iteritems():
            for destination_id in xpath_counts_dict[key]:
                xpath_counts_dict[key][destination_id] /= count
                print key, destination_id, xpath_counts_dict[key][destination_id]
        '''
        self.navigation_table = defaultdict(lambda : defaultdict(float))
        self.average_link_table =  defaultdict(lambda : defaultdict(float))
        self.counts_dict = counts_dict
        for key,destination_list in xpath_counts_dict.iteritems():
            cluster_id, xpath = key[0], key[1]
            norm = sum(destination_list.values())
            for destination_id in destination_list:
                self.navigation_table[key][destination_id] = xpath_counts_dict[key][destination_id]/float(norm)
                self.average_link_table[key][destination_id] = xpath_counts_dict[key][destination_id]/counts_dict[cluster_id]

        #entropy_score = self.entropy(self.navigation_table)
        #print "Micro average entropy is " + str(entropy_score)
        #entropy_file = open("entropy.out","a")
        #entropy_file.write("Micro average entroy of {0} is {1}\n".format(self.dataset,entropy_score))
        #raise#entropy

        ''' output
        for key in xpath_counts_dict:
            if key[0] == 1:
                print key, xpath_counts_dict[key]
        '''
        print "=========== end of training ============"
        self.xpath_counts_dict = xpath_counts_dict
        self.trans_dict = trans_dict
        print self.xpath_counts_dict, "xpath count dict"

        for key in self.navigation_table:
            if int(key[0]) == 6:
                print key, self.navigation_table[key], "navigational table"
        return xpath_counts_dict

    def get_trans_dict(self,dataset,date):
        s = sampler(self.dataset,self.entry,self.prefix,0)
        folder = "../../Crawler/{}_samples/{}".format(date,dataset)
        s.folder = folder
        print folder
        for file in os.listdir(folder):
            print file
            s.crawl_link(file,set())
        #print s.transition_dict, "s.transition_dict"
        return s.transition_dict

    # @ input:  self.navigation_table[(cluster_id, xpath)] -> probability distribution
    # @ output: first the transition probability for each page and then we generate the transition matrix on cluster/cluster level
    def get_transition_matrix(self):
        transition_mat = defaultdict(lambda : defaultdict(float))
        trans_dict = self.trans_dict  # [page][xpath] = [url list] ->[cluster][xpath] = {probability list}
        page_count = 0
        cluster_trans_prob = np.zeros((self.cluster_num,1))
        for page_path, trans in trans_dict.iteritems():
            page_url = page_path.replace(".html","").replace("_","/")
            if page_url not in self.gold_dict:
                continue
            else:
                page_count += 1
                cluster_id = self.cluster_dict[page_url]

            for xpath,url_list in trans.iteritems():
                length = len(url_list)
                if length <= 0:
                    continue
                key = (cluster_id,xpath)
                distribution = self.navigation_table[key]

                for dest_id, prob in distribution.iteritems():
                    print page_url, key, dest_id, prob, length, url_list[0][0]
                    cluster_trans_prob[dest_id][0] += prob*length # expected outlink to a given cluster
            print cluster_trans_prob.T
            cluster_trans_prob = normalize(cluster_trans_prob,norm="l1",axis=0) # normalize!
            print cluster_trans_prob.T
            for i in range(self.cluster_num):
                transition_mat[cluster_id][i] += cluster_trans_prob[i][0]


        print "number with prediction is {}".format(page_count)
        print transition_mat
        return transition_mat


    def get_adjacency_matrix(self):
        adjacency_mat = defaultdict(lambda : defaultdict(float))
        #for key, distribution in self.average_link_table.iteritems():
        for key, distribution in self.xpath_counts_dict.iteritems():
            cluster_id, xpath = key[0], key[1]
            for dest_id, num_links in distribution.iteritems():
                adjacency_mat[cluster_id][dest_id] += num_links
        print "adjacency mat", adjacency_mat
        return adjacency_mat






    # evaluate the entropy of xpath estination
    # micro output average for each xpath
    def entropy(self,xpath_counts_dict):
        entropy_list = []
        print xpath_counts_dict, "jajaja"
        for key, distribution in xpath_counts_dict.iteritems():

            count_sum = sum(distribution.values())
            prob_list = []
            for cluster in distribution:
                prob_list.append(distribution[cluster]/count_sum)
            #print prob_list
            entropy = 0.0
            for prob in prob_list:
                entropy += prob * math.log(prob,2)
            entropy_list.append(entropy)
        micro_entropy =  (-1) * sum(entropy_list)/len(entropy_list)
        return micro_entropy # the lower, the better


    # @ input: self.sitemap
    # @ output: two dict  : mapping url -> class/cluster
    # @ return: self.gold_dict & self.cluster_dict
    # @ need to stem the path, since it is a path instead of url
    def build_gold_cluster_dict(self):
        pages = self.sitemap.UP_pages
        self.gold_dict, self.cluster_dict = {},{}
        for index, path in enumerate(pages.path_list):
            #print path, "url in pages.path_list"


            url = path.replace(self.path_prefix.replace("_","/"),"")
            #print self.path_prefix
            #print url, "building gold,cluster dict"
            self.gold_dict[url] = pages.ground_truth[index]
            self.cluster_dict[url] = pages.category[index]


    # input: test_set, target_cluster, trans_xpath_dict: [cluster,xpath]-> [cluster_id] ->num
    # prediction results
    def predict_destination(self,file_path,target_cluster,rules):
        #print self.cluster_xpath_trans
        page,cluster_id = self.classify(file_path)
        print rules

        if self.date == "May1":
            link_dict = page.getAnchor(True)
        else:
            link_dict = page.getAnchor()

        right,guess,total,miss = 0,0,0,0

        for xpath in link_dict:
            gold_list = []
            #print self.cluster_xpath_trans[(cluster_id,xpath)]
            if target_cluster in  self.cluster_xpath_trans[(cluster_id,xpath)]:
                tmp = self.cluster_xpath_trans[(cluster_id,xpath)]
                print tmp, (cluster_id,xpath)
                # we only focus on intra links
                link_list = []
                for link in link_dict[xpath]:
                    if self.intraJudge(link,self.dataset)>0:
                        link_list.append(link)
                total += len(link_list)
                guess += len(link_list) * (tmp[target_cluster]/sum(tmp.values()))

                print xpath,(tmp[target_cluster]/sum(tmp.values())), link_list
                for link in link_list:
                    #if "playlist?" in link:
                    #if self.match_rules(link,[]):
                    #if self.match_rules(link,[]):
                    #if self.match_rules(link,[["p","^[0-9]+$"],["post","^[0-9]+.aspx$"],["t","^[0-9]+.aspx(.*)$"],["t","next","^[0-9]+$"],["t","prev","^[0-9]+$"]]):
                    #if self.fmatch_rules(link,[["members"]]):
                    #if self.match_rules(link,[["^Hotel_Review-(.*)$"]]):
                    if self.match_rules(link,rules):
                    #if self.match_rules(link,[["subject","^[0-9]+$","questions","ask"]]):
                        gold_list.append(link)
                        right += 1
                print link_list
                print gold_list
            else:
                link_list = []
                for link in link_dict[xpath]:
                    if self.intraJudge(link,self.dataset)>0:
                        link_list.append(link)
                for link in link_list:
                    if self.match_rules(link,rules):
                        miss += 1

        if right !=0:
            print right,guess
            precision = 1 - math.fabs(right-guess)/total
            print link_dict
        else:
            precision = None
        if miss+right ==0:
            recall = None
        else:
            recall = float(right)/float(miss+right)

        print  precision, " 1-|right - guess|/total"
        print  recall, "|right|/|right+miss|"
        return cluster_id,precision,recall


    def get_rules(self):
        if self.dataset=="stackexchange":
            cluster_list = [[["a","^[0-9]+$"],["q","^[0-9]+$"],["questions","^[0-9]+$"]],\
                            [["users","^[0-9]+(.*)$"]], [["^search?(.*)$"]]]
            rules = cluster_list[self.cluster_rank]
        elif self.dataset == "youtube":
            cluster_list =[[["^(.*)watch?v=(.*)$"]],[["^(.*)playlist?(.*)$"]],[["user","^discussion(.*).html$"]]]
            rules = cluster_list[self.cluster_rank]
        elif self.dataset == "asp":
            cluster_list =[[["p","^[0-9]+$"],["post","^[0-9]+.aspx$"],["t","^[0-9]+.aspx(.*)$"],["t","next","^[0-9]+$"],["t","prev","^[0-9]+$"]],\
            [["^members(.*)$"]],[["login","^RedirectToLogin?(.*)$"]]]
            rules = cluster_list[self.cluster_rank]
        elif self.dataset == "tripadvisor":
            cluster_list = [[["^Hotel_Review-(.*)$"]],\
                            [["^ShowUserReviews-(.*)$"]]]
            rules = cluster_list[self.cluster_rank]
        elif self.dataset == "douban":
            cluster_list = [[["review","^[0-9]+$"]],\
                            [["subject","^[0-9]+$","questions","ask"]]]
            rules = cluster_list[self.cluster_rank]
        print rules
        return rules


    def intraJudge(self,url, site):
    # oulink with http or symbol like # and /
        if len(url)==0:
            return 0

        if site == "stackexchange":
            if url[0]=="/" and url[0:2] !="//":
                return 1
            else:
                if "http://android.stackexchange.com/" in url:
                    return 2
                else:
                    return 0
        elif site == "yelp":
            if len(url) == 1 or "http" in url:
                if "http://www.yelp.com" in url:
                    return 0
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                print url
                return 1
        elif site == "asp":
            if len(url) == 1 or "http" in url:
                if "http://forums.asp.net" in url:
                    return 0
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                if url[0] == "/":
                    return 1
                else:
                    return 0
        elif site == "douban":
            if "http" in url:
                if "movie.douban.com" in url:
                    return 2
                else:
                    return 0
            else:
                return 0
        elif site == "tripadvisor":
            if "http" in url:
                if "tripadvisor.com" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                return 1
        elif site == "hupu":
            if "http" in url:
                if "voice.hupu.com" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                return 1
        elif site == "biketo":
            if "http" in url:
                if "biketo.com" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//" and not url.endswith(".jpg"):
                return 1
            else:
                return 0
        elif site == "amazon":
            if "http" in url:
                return 0
            elif url[0:2] == "//":
                return 0
            else:
                return 1
        elif site == "youtube":
            if "https://www.youtube.com" in url:
                return 2
            elif url[0:2] == "//":
                return 0
            else:
                if url[0:1] == "/":
                    return 1
                else:
                    return 0
        elif site == "csdn":  # http://bbs.csdn.net/home # http://bbs.csdn.net
            if "http" in url:
                if "my.csdn.net" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                if "javascript:void(0)" in url:
                    return 0
                else:
                    return 1
        elif site == "baidu":
            if "http" in url:
                if "tieba.baidu.com" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                if "javascript:void(0)" in url:
                    return 0
                else:
                    return 1
        elif site == "huffingtonpost":
            if "http" in url:
                if "http://www.huffingtonpost.com/" in url and not url.endswith(".jpg"):
                    return 2
                else:
                    return 0
        elif site == "rottentomatoes":
            if "http" in url:
                if url.startswith("https://www.rottentomatoes.com"):
                    return 2
            elif url[0:2] == "//":
                return 0
            else:
                if url[0:1] == "/":
                    return 1
                else:
                    return 0
        else:
            return 0

    def get_sample_cluster(self):
        print "predicting"
        '''
        print self.sitemap.pre_y
        c = Counter(self.sitemap.pre_y)
        #print c, "counter of pred_y"
        temp = sorted(c.iteritems(),key=lambda x:x[1],reverse=True)
        print temp, "sorted cluster for sitemap"
        cluster_id = temp[self.cluster_rank][0]
        return int(cluster_id)
        '''
        #print "The {0} th largest cluster id is {1}".format(self.cluster_rank,self.target_cluster)
        #raise

        # The below is for the first version
        if self.dataset=="youtube":
            cluster_list = ["../../Crawler/May1_samples/youtube/https:__www.youtube.com_watch?v=zhnMSVb0oYA.html",\
                            "../../Crawler/July30_samples/youtube/https:__www.youtube.com_user_RoosterTeeth.html",\
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
                            "../../Crawler/July30_samples/douban/https:__movie.douban.com_subject_3541415_.html",\
                            "../../Crawler/July30_samples/douban/https:__movie.douban.com_celebrity_1303681_.html"]
            file_path = cluster_list[self.cluster_rank]
        elif self.dataset == "hupu":
            cluster_list = ["../../Crawler/July30_samples/hupu/http:__voice.hupu.com_nba_1482921.html.html",\
                            "../../Crawler/July30_samples/hupu/http:__voice.hupu.com_people_4915254.html"]
            file_path = cluster_list[self.cluster_rank]
        elif self.dataset == "rottentomatoes":
            cluster_list = ["../../Crawler/July30_samples/rottentomatoes/http:__www.rottentomatoes.com_m_macbeth_2015_.html",\
                            "../../Crawler/July30_samples/rottentomatoes/http:__www.rottentomatoes.com_celebrity_kenny_loggins.html"]
            file_path = cluster_list[self.cluster_rank]
        #elif self.dataset == "":
        #    cluster_list = []


        page,cluster_id = self.classify(file_path)
        self.test_page = file_path
        print "file path is ", file_path
        print "we foucs on cluster {0} for {1}".format(cluster_id,self.dataset)
        return int(cluster_id)

    def match_rules(self,url,rule_list):
        for rule in rule_list:
            if self.match(url,rule):
                return True


    def match(self, url, rule):
        strip_url = url.strip()
        temp, terms = strip_url.split("/"), []
        for term in temp:
            if term != "":
                terms.append(term)
        match_id = 0
        for index,term in enumerate(terms):
            if rule[match_id][0]=="^" and rule[match_id][-1] == "$":
                try:
                    if re.match(rule[match_id],term):
                        match_id += 1
                except:
                    traceback.print_exc()
                    print rule[match_id]
            else:
                if term == rule[match_id]:
                    match_id += 1
            if match_id >= len(rule):
                break

        if match_id >= len(rule):
            return True
        else:
            return False

    def update_cluster_walk_list(self,url,cluster_id):
        print url, cluster_id, "in updateing walk list", type(cluster_id)
        print self.walk_list,"oh walk list"
        if cluster_id == -1:
            self.walk_list[self.cluster_num] += 1
        else:
            self.walk_list[cluster_id] += 1



    def sampling(self,num_crawl,method="uniform"):
        # need to read the pagerank dict from file
        if method == "pagerank":
            path = "./src/data/{0}/{0}.pr_dict".format(self.dataset)
            with open(path,"rb") as outfile:
                pr_dict = pickle.load(outfile)
            avg_pr = sum(pr_dict.values())/len(pr_dict)
            print avg_pr, "average pagerank"
        elif method == "indegree":
            path = "./src/data/{0}/{0}.inlink_dict".format(self.dataset)
            with open(path,"rb") as outfile:
                inlink = pickle.load(outfile)
            indegree_dict = defaultdict(int)
            for key in inlink:
                indegree_dict[key] = len(inlink[key])
            avg_indegree = sum(indegree_dict.values())/len(indegree_dict)
        elif method == "est_prob":
            counter = Counter(sitemap.UP_pages.category)
            self.c_prob = defaultdict(float)
            total = sum(counter.values())
            for key in counter:
                self.c_prob[key] = float(counter[key])/float(total)

        self.crawl_history = Counter()
        for i in range(self.cluster_num):
            self.crawl_history[i] = 1
        print "file path {}".format("./results/sampling/sampling_{0}_{1}_{2}_size{3}.txt".format(method,self.dataset,self.date,self.crawl_size))
        write_file = open("./results/sampling/sampling_{0}_{1}_{2}_size{3}.txt".format(method,self.dataset,self.date, self.crawl_size),"w")
        num_web_crawl=0
        entry, prefix = self.entry, self.prefix
        self.url_stack,self.crawl_length, self.walk_list = [entry],0,defaultdict(int)
        self.final_list, url_list, last_list = [], [], []
        size, num = num_crawl,  0 # the number of crawling
        s = sampler(self.dataset,self.entry,self.prefix,0)
        while(num<size and len(self.url_stack) >0):
            first_url = self.url_stack[0]
            try:
                print "first_url", first_url
            except:
                print "name error"
            try:
                sys.stdout.write("num is {}\n".format(num))
                sys.stdout.flush()
                #print num, "num"
                url_list,cluster_id = self.sample_link(first_url,s,method)
                #if first_url not in self.history_set:
                    # i think since the first url has not been poped out , it will sampled again .. but what if all of
                    # pages links has been sampled ? it will go into a infinite loop..

                # add url to sample history anyway
                self.crawl_history[cluster_id] += 1
                self.crawl_length += 1
                if method == "est_ratio":
                    self.update_cluster_walk_list(first_url,cluster_id)

                if first_url not in self.history_stack:
                    num += 1
                self.final_list.append((first_url,cluster_id))

            except:
                print "might miss somthing here"
                traceback.print_exc()
                flag = s.crawlUrl(first_url,self.dataset,self.url_stack,self.history_stack)
                if flag == 1:
                    sys.stdout.write("num is {}\n".format(num))
                    sys.stdout.flush()
                    #print num, "num"
                    url_list,cluster_id = self.sample_link(first_url,s,method)
                    if method == "est_ratio":
                        self.update_cluster_walk_list(first_url,cluster_id)
                    print "url_list", url_list
                    if first_url not in self.history_stack:
                        num += 1
                    self.final_list.append((first_url,cluster_id))

                    random_time_s = random.randint(5, 10)
                    time.sleep(random_time_s)
                    #num_web_crawl += 1
                    if num_web_crawl%10 == 9:
                        random_time_s = random.randint(60, 90)
                        time.sleep(random_time_s)
                else:
                    #change the first_url from parent sampling
                    print num, "num"
                    traceback.print_exc()
                    pass

            if self.url_stack[0] == first_url:
                self.url_stack.pop(0)
                if first_url not in self.history_stack:
                    self.history_stack.append(first_url)

            probability = 0.15
            self_pr = self.pr_score[cluster_id][0]/self.num_mat[cluster_id][0]
            if method=="uniform":
                # after processing, 0.15 random and 0.85 uniform sampling
                # no out-links - random sample
                if random.random() < probability:
                    self.select_from_history_stack()
                    print "random sampled from history set"
                else:
                    try:
                        print url_list
                        id = random.randrange(len(url_list))
                        self.url_stack.append(url_list[id])
                        print url_list[id], "select from out-links"
                    except:
                        self.select_from_history_stack()

            elif method == "pagerank":
                if random.random() < probability:
                    url = random.sample(self.history_stack,1)[0]
                    self.url_stack.append(url)
                    print url, "random sampled from history set"
                else:
                    try:
                        id = self.sample_from_dist(url_list,pr_dict,avg_pr)
                        self.url_stack.append(url_list[id])
                        print url_list[id], "select from out-links"
                    except:
                        traceback.print_exc()
                        self.select_from_history_stack()
            elif method == "indegree":
                print "sample from orcacle indegree"
                if random.random() < probability:
                    url = random.sample(self.history_stack,1)[0]
                    self.url_stack.append(url)
                    print url, "random sampled from history set"
                else:
                    try:
                        id = self.sample_from_dist(url_list,indegree_dict,avg_indegree)
                        self.url_stack.append(url_list[id])
                        print url_list[id], "select from out-links"
                    except:
                        traceback.print_exc()
                        self.select_from_history_stack()

            elif method == "visit_ratio":
                print "using the visit ratio during the walk to estimate the probability"
                if random.random() < probability:
                    url = random.sample(self.history_stack,1)[0]
                    self.url_stack.append(url)
                    print url, "random sampled from history stack because of the 0.20 probability"
                    self.walk_list[url] += 1
                else:
                    try:
                        avg_count = 1
                        print avg_count, " average count is "
                        id = self.sample_from_dist(url_list,self.walk_list,avg_count)
                        self.url_stack.append(url_list[id])
                        self.walk_list[url_list[id]] += 1
                        print url_list[id], "select from out-links"
                    except:
                        traceback.print_exc()
                        self.select_from_history_stack()



            else: # our method
                print "our method"
                if random.random() < probability:
                    url = random.sample(self.history_stack,1)[0]
                    self.url_stack.append(url)
                    print url, "random sampled from history set"
                else:
                    try:
                        id = self.sample_from_prob_list(url_list,self_pr)
                        self.url_stack.append(url_list[id][0])
                        print url_list[id], "select from out-links"
                    except:
                        traceback.print_exc()
                        self.select_from_history_stack()


        print len(self.final_list), "length of final list"
        for pair in self.final_list:
            url, cluster_id = pair[0],pair[1]
            write_file.write(url + "\t"+ str(cluster_id) + '\n')
                #random sample one from url list and add to ur_stack

    def select_from_history_stack(self):
        url = random.sample(self.history_stack,1)[0]
        self.url_stack.append(url)
        print url, "random sampled from history stack"


    # input: entry, prefix, site, trans_xpath_dict , target_cluster id
    # target: find the cluster page that we want
    def crawling(self,num_crawl):

        # self.entry, self.prefix, self.dataset, self.trans_xpath_dict, target_cluste id
        #self.target_cluster = self.get_sample_cluster()
        write_file = open("./results/{0}_{1}_{2}_{3}_sitemap{4}_size{5}.txt".format(self.dataset,self.date, self.cluster_rank, self.rank_algo, self.num_samples, self.crawl_size),"w")
        num_web_crawl = 0
        entry,prefix = self.entry, self.prefix
        self.url_stack  = [(entry,"","",self.max_score)]
        self.final_list = []
        size, num = num_crawl,  0 # the number of crawling
        crawl_id = 0
        s = sampler(self.dataset,self.entry,self.prefix,0)
        while(num<size and len(self.url_stack)>0):
            first_url = self.url_stack[0][0]
            parent_url = self.url_stack[0][1]
            parent_xpath = self.url_stack[0][2]
            score = self.url_stack[0][3]
            #print self.url_stack[0],"first element"
            #print self.url_stack[-1], "last element"

            #first_url = self.url_stack[0][0]
            try:
                print "first url is ",first_url

            except:
                traceback.print_exc()
            if first_url not in self.history_set:
                num += 1
                try:
                    url_list,cluster_id = self.crawl_link(first_url,  self.history_set, s)
                    flag = self.rank_algo == "vidal" and cluster_id == self.target_cluster
                    if not flag:
                        self.sort_queue(url_list,first_url,self.rank_algo) # sort url_stack
                    self.final_list.append((first_url,parent_url,parent_xpath,score,cluster_id))
                except:
                    print "might miss somthing here"
                    traceback.print_exc()
                    flag = s.crawlUrl(first_url,self.dataset,self.url_stack,self.history_set)
                    if flag == 1:
                        url_list,cluster_id = self.crawl_link(first_url,  self.history_set,s )
                        flag = self.rank_algo == "vidal" and cluster_id == self.target_cluster
                        if not flag:
                            self.sort_queue(url_list,first_url,rank_algo=self.rank_algo)
                        self.final_list.append((first_url,parent_url,parent_xpath,score,cluster_id))
                        random_time_s = random.randint(5, 10)
                        time.sleep(random_time_s)
                        num_web_crawl += 1
                        if num_web_crawl%15 == 14:
                            random_time_s = random.randint(45, 90)
                            time.sleep(random_time_s)
                    else:
                        num -= 1
                        print "crawl failure"
            if self.url_stack[0][0] == first_url:
                self.url_stack.pop(0)
            crawl_id += 1
            print " num is {}".format(num)
            sys.stdout.flush()
            if num >= size:
                print "crawl_id is {0} for size {1}".format(crawl_id,size)

                #print "first url comes from the {} th crawled page".format(self.page_num[first_url])
            self.history_set.add(first_url)
        print len(self.final_list), "length of final list"
        for pair in self.final_list:
            url, parent_url,parent_xpath,score, cluster_id = pair[0],pair[1],pair[2],pair[3],pair[4]
            write_file.write(url + "\t" + str(parent_url) +"\t" + str(parent_xpath) + "\t" + str(score) +  "\t"+ str(cluster_id) + '\n')

    def sort_queue(self,url_list,first_url,rank_algo):

        for url in url_list:
            #print url, "an instance in url_list"
            if url[0] not in self.history_set and url not in self.url_stack:
                self.url_stack += [url]

        #self.url_stack.pop(0) # delete current
        if rank_algo == "bfs" or rank_algo == "vidal":
            print "bfs for crawling frontier"
            #time.sleep(10)
            print self.url_stack[0], "first element in url stack"
        elif rank_algo == "target":
            print "algo is sorting by hub and authority"
            print len(self.url_stack), "length of url_stack"
            self.url_stack = sorted(self.url_stack,key=lambda tup:tup[-1],reverse=True)  # sorts in place
            #print "===" + str(self.url_stack) + "==="
            '''
            for url in self.url_stack:
                print url[0],url[-1]
            raise
            '''
        # general - 3 terms / no_sim : link + balance / no_balance:
        elif rank_algo == "general" or rank_algo == "no_sim" or rank_algo == "no_balance" or rank_algo=="info" or rank_algo=="sim":
            print "general crawling which takes the ratio of clusters into consideration"
            print len(self.url_stack), "length of url_stack"
            self.url_stack = sorted(self.url_stack,key=lambda tup:tup[-1],reverse=True)  # sorts in place

        else:
            raise
        if len(self.url_stack)>3000:
            self.url_stack = self.url_stack[:3000]



    def test_destination(self):
            write_file = open("./evaluation/predict_destination.txt","a")
            target_cluster = self.get_sample_cluster() #
            rules = self.get_rules()
            folder_path = "../../Crawler/Mar15_samples/{}/".format(args.dataset)
            macro_precision,macro_recall, num, num_recall = 0.0,0.0, 0,0
            test_cluster_list = []

            for file in os.listdir(folder_path):
                file_path = folder_path + file
                print file_path, "for test set Mar 15"
                cluster_id, precision, recall = self.predict_destination(file_path,target_cluster,rules)
                test_cluster_list.append(cluster_id)
                if precision is not None:
                    macro_precision += precision
                    num += 1
                if recall is not None:
                    macro_recall += recall
                    num_recall += 1

            macro_precision = float(macro_precision)/float(num)
            macro_recall = float(macro_recall)/float(num_recall)
            print macro_precision, " macro average of precision"
            print macro_recall, " macro average of recall"

            write_file.write(self.dataset + "\t" + self.date + "\t" + "rank:{}\t".format(self.cluster_rank) + "\t" + "macro_precision:{}".format(macro_precision) +
                             "\t" + "macro_recall:{}".format(macro_recall) + "\n")

            c = Counter(test_cluster_list)
            print c, "counter"

    def crawl_link(self, first_url, history_stack, sampler):
        file_path = self.full_folder + "/" + first_url.replace("/","_") +".html"
        #print file_path
        page,cluster_id = self.classify(file_path)
        self.crawled_cluster_count[cluster_id] += 1
        available_url_list = []
        print file_path, cluster_id , "file_path for cluster_id"
        original = open(file_path,"r").read()
        contents = original.replace("\n","")
        link_dict = sampler.getAnchor(contents,first_url,sample_flag=False)

        #print link_dict, "link dict"

        #self.transition_dict[url] = link_dict
        for xpath in link_dict:
            #print xpath,"xpath is now"
            # considering cluster
            #distribution = self.cluster_xpath_trans[(cluster_id,xpath)]
            distribution = self.xpath_counts_dict[(cluster_id,xpath)]
            print distribution, (cluster_id,xpath), " cluster_id, xpath for distribution"
            score = self.calculate_url_importance(distribution,self.rank_algo) # and self.cluster_trans_prob_mat
            #print distribution, "the probability of itself"
            link_list = link_dict[xpath]
            for url in link_list:
                if self.judge_in_corpus(url):
                    print "url is ", url, score
                    if url not in history_stack and url not in available_url_list:
                        available_url_list.append((url,first_url,(cluster_id,xpath),score))
        return available_url_list, cluster_id

    def sample_link(self,first_url,sampler,method="uniform"):
        #print "sample link starts"
        file_path = self.full_folder + "/" + first_url.replace("/","_") +".html"
        page, cluster_id = self.classify(file_path)
        original = open(file_path,"r").read()
        contents = original.replace("\n","")
        link_dict = sampler.getAnchor(contents,first_url,sample_flag=False)
        available_url_list = []
        #print self.xpath_counts_dict, "xpath_counts_dict"
        if method == "uniform" or method == "pagerank" or method == "indegree" or method == "visit_ratio":
            print "no predicting scores"
            for xpath in link_dict:
                link_list = link_dict[xpath]
                for link in link_list:
                    if link not in self.history_set:
                        available_url_list.append(link)
        else:
            print "predicting scores based on distribution"
            for xpath in link_dict:
                #distribution = self.xpath_counts_dict[(cluster_id,xpath)]
                distribution = self.navigation_table[(cluster_id,xpath)]
                #print distribution, " distribution is "
                link_list = link_dict[xpath]
                if method == "est_pagerank":
                    score = self.calculate_url_pr(distribution,cluster_id)
                elif method == "est_degree":
                    score = self.calculate_url_indegree(distribution)
                elif method == "est_ratio":
                    score = self.calculate_url_prob(distribution)
                #elif method == "est_ratio":
                #    score = self.calculate_url_prob()
                #print score, "calcualte_url_pr"
                for link in link_list:
                    if link not in self.history_set:
                        available_url_list.append((link,score))
        #print "sample link ends"
        return available_url_list, cluster_id

    # input: url_list and its pagerank score distribution dict (key->value)
    # output: an id sampled inversed to its pagerank score
    def sample_from_dist(self,url_list,pr_dict,avg_pr,self_pr):
        pr_dist = []
        outlier_list = []
        out_degree = len(url_list)
        print avg_pr, "average pagerank"
        #print url_list
        for id,url in enumerate(url_list):
            if url in pr_dict:
                if pr_dict[url] !=0:
                    pr_dist.append(min(self_pr/pr_dict[url],1.0)/float(out_degree))
                else:
                    pr_dist.append(1.0/float(out_degree))
                    outlier_list.append(id)
            else:
                outlier_list.append(id)
                pr_dist.append(1.0/float(out_degree))
        pr_dist = normalize(pr_dist,"l1")[0]
        print len(pr_dist),
        print pr_dist, "pagerank/indegree list"

        if len(pr_dist) == 0:
            raise Exception("Has no links")

        cdf = [pr_dist[0]]
        for i in xrange(1, len(pr_dist)):
            cdf.append(cdf[-1] + pr_dist[i])
        random_ind = bisect(cdf,random.random())
        print random_ind, "random sampling id from pagerank score "
        if random_ind in outlier_list:
            print "woca outlier!"
        return random_ind


    # input, list of pairs (url,score)
    def sample_from_prob_list(self,url_list,self_pr):
        pr_dist = []
        self_link = 0
        for item in url_list:
            if item[-1] == -1:
                self_link += 1
        out_degree = len(url_list) - self_link

        PMH = 0.0
        for item in url_list:
            pr = item[-1]
            if pr != -1:
                PMH += min(self_pr/pr,1.0)/float(out_degree)
        self_prob = (1- PMH)/self_link
        for item in url_list:
            pr = item[-1]
            if pr != -1:
                pr_dist.append(min(self_pr/pr,1.0)/float(out_degree))
            else:
                pr_dist.append(self_prob)

        print pr_dist, " reversed prob list"
        pr_dist = normalize(pr_dist,"l1")[0]
        #print url_list, " estimated pagerank prob distribution"
        print pr_dist, " reversed probability distribution"
        if len(pr_dist) == 0:
            raise Exception("Has no links")

        cdf = [pr_dist[0]]
        for i in xrange(1, len(pr_dist)):
            cdf.append(cdf[-1] + pr_dist[i])
        random_ind = bisect(cdf,random.random())
        print random_ind, pr_dist[random_ind],"random sampling id from pagerank score "
        return random_ind


    # @ input: cluster_xpath_trans (cluster,xpath) -> distribution
    # @ output:  self.trans_prob_mat, [cluster_id][cluster_id] -> n
    def calculate_trans_prob_mat(self):
        trans_prob_mat = defaultdict(lambda : defaultdict(float))
        for pair, distribution in self.cluster_xpath_trans.iteritems():
            start_id, xpath = pair[0], pair[1]
            for dest_id,num in distribution.iteritems():
                trans_prob_mat[start_id][dest_id] += num
                #if dest_id == 9 and num > 10:
                #    print num, pair , dest_id, " why this path is so powerful!!!"
        self.debug_file.write(str(trans_prob_mat))
        print trans_prob_mat, "tans_prob_mat"

        #self.cluster_num = max(trans_prob_mat.keys())+1
        return trans_prob_mat

    def calculate_url_importance(self,distribution,rank_algo="general"):
        if rank_algo == "general" or rank_algo == "no_sim" or rank_algo == "no_balance" or rank_algo=="info" or rank_algo=="sim":
            k1, k2, k3, k4 = 0.5, 0.5, 0.0, 0.0
        elif rank_algo == "target":
            k1, k2, k3, k4 = 0.8, 0.2, 0.0, 0.0
        elif rank_algo == "irobot":
            k1, k2, k3, k4 = 0.0, 0.0, 1.0, 0.0
        elif rank_algo == "bfs" or rank_algo=="vidal":
            return 0.0
        score = 0.0
        total = sum(self.crawled_cluster_count.values()) - self.crawled_cluster_count[-1] + 1
        norm = sum(distribution.values())
        # -1?
        for dest_id, num in distribution.iteritems():
            if dest_id == -1:
                continue
            prob = float(num)/float(norm)
            if k3 == 1.0:
                score += prob * self.sitemap.cluster_importance[dest_id]
            else:
                #discount = (1-float(self.crawled_cluster_count[dest_id])/float(total))
                ratio = float(self.crawled_cluster_count[dest_id])/float(total)
                ratio_weight = 1 - ratio
                #print total, "total", discount, "discount"
                tmp = 0.0
                tmp += k1* prob * self.auth_score[dest_id]
                tmp += k2* prob * self.hub_score[dest_id]
                if rank_algo == "general":
                    ratio_weight =  1.0 * ratio_weight
                    tmp *= self.sitemap.intra_similarity[dest_id]
                    #print "intra dist_similarity is " + str(self.sitemap.intra_similarity[dest_id])
                    #print ratio_weight, k3,"ratio weight"
                    #score += tmp * self.c_prob[dest_id]
                    tmp *= ratio_weight
                    score += tmp
                elif rank_algo == "no_sim":
                    ratio_weight =  1.0 * ratio_weight
                    tmp *= ratio_weight
                    score += tmp
                elif rank_algo == "no_balance":
                    tmp *= self.sitemap.intra_similarity[dest_id]
                    score += tmp
                elif rank_algo == "info":
                    score += tmp
                elif rank_algo == "sim":
                    print "sim metric"
                    score += prob * self.sitemap.intra_similarity[dest_id] * ratio_weight
                # target
                else:
                    score += tmp

        return score

    def calculate_url_pr(self,distribution,cluster_id):
        #print distribution, " distribution "
        if len(distribution) == 0:
            return 1

        score = 0.0
        guess_id = self.guess_cluster_id(distribution)
        #score, norm = 0.0, sum(distribution.values())
        #print distribution, "distribution!"
        #for dest_id, prob in distribution.iteritems():
            #print dest_id, prob
            #prob = float(num)/float(norm)
            #if dest_id == -1:
                #score += prob * self.avg_pr_score[0]
                #score += prob * self.pr_score[self.cluster_num-1][0]/self.num_mat[self.cluster_num-1][0]
                #print dest_id, self.avg_pr_score
            #else:
            #score += prob * self.pr_score[dest_id][0]/self.num_mat[dest_id][0]
                #print "dest_id", dest_id, self.pr_score[dest_id][0]
            #print self.pr_score[dest_id][0]

        if guess_id == cluster_id:
            score = -1
        else:
            score = self.pr_score[guess_id][0]/self.num_mat[guess_id][0]
        print "distribution is ", distribution, guess_id, score
        return score

    def calculate_url_indegree(self,distribution):
        if len(distribution) == 0:
            return self.avg_indegree
        score, norm = 0.0, sum(distribution.values())
        for dest_id, num in distribution.iteritems():
            prob = float(num)/float(norm)
            score += prob * self.indegree[dest_id]
            #print self.pr_score[dest_id][0]
        return score

    def calculate_url_prob(self,distribution,cluster_id):
        # self.crawl_history
        if len(distribution) == 0:
            #return 1.0/float(sum(self.crawl_history.values()))
            print "no data record"
            return 1.0/float(self.cluster_num)
        #c_prob = defaultdict(float)
        #print self.crawl_history, "crawl history "
        #for key,value in self.crawl_history.iteritems():
        #    c_prob[key] = float(value)/float(sum(self.crawl_history.values()))
        #print c_prob, "c_prob"
        print distribution, "distribution"
        print self.walk_list, "walk list"
        total = sum(self.walk_list.values())
        print total, "total number of walk"
        score, norm = 0.0, sum(distribution.values())
        '''
        for dest_id, num in distribution.iteritems():
            prob = float(num)/float(norm)
            if dest_id == -1:
                print max(self.walk_list[self.cluster_num],1)/max(total,1), "oh for outlier prob"
                score +=  prob*max(self.walk_list[self.cluster_num],1)/max(total,1)
            else:
                score += prob * max(self.walk_list[dest_id],1)/max(total,1)
        '''
        guess_id = self.guess_cluster_id(distribution)
        if guess_id == cluster_id:
            score = -1
        else:
            score = self.pr_score[guess_id]/self
        return score

    def guess_cluster_id(self,distribution):
        max_id = max(distribution.keys())
        cdf = [distribution[0]]
        for i in xrange(1, max_id):
            cdf.append(cdf[-1] + distribution[i])
        random_ind = bisect(cdf,random.random())
        return random_ind

    def compute_hit_scores(self,max_iter=100):
        trans_mat = self.trans_mat

        auth_score = np.ones((self.cluster_num, 1)) / float(self.cluster_num)
        hub_score = np.ones((self.cluster_num, 1)) / float(self.cluster_num)
        # previous_pr = np.zeros((doc_num,1))
        ite = 0
        while (ite < max_iter):
            # previous_pr = pr_score
            hub_score = trans_mat.dot(auth_score)
            auth_score = trans_mat.T.dot(hub_score)
            auth_score = normalize(auth_score, norm='l1', axis=0)
            hub_score = normalize(hub_score, norm='l1', axis=0)
            ite += 1
        self.auth_score = auth_score.T[0]
        self.hub_score = hub_score.T[0]
        self.output_hits()  # n*1

    def compute_pagerank_scores(self,max_iter=100):
        #self.cluster_num += 1
        alpha = 0.80
        trans_mat = self.trans_mat
        trans_mat = normalize(trans_mat, norm='l1', axis=0)
        print trans_mat.T, "trans_mat transpose"
        pr_score = np.ones((self.cluster_num,1))/float(self.cluster_num)
        max_id = max(self.sitemap.pre_y)
        print "max id and cluster num is ", max_id, self.cluster_num
        teleportion_mat,num_mat = np.zeros((self.cluster_num,1)),np.zeros((self.cluster_num,1))
        c = Counter(self.sitemap.pre_y)
        denominator = 0.0
        print c
        print self.cluster_num
        for i in range(self.cluster_num):
            if i == self.cluster_num -1:
                teleportion_mat[i],num_mat[i] = c[-1],c[-1]
                denominator += c[-1]
            else:
                teleportion_mat[i],num_mat[i] = c[i],c[i]
                denominator += c[i]
        teleportion_mat = teleportion_mat/denominator
        print teleportion_mat, " teleportation matrix"
        ite = 0
        while(ite < max_iter):
            #print "iteration ", ite
            #print trans_mat, trans_mat.shape
            #print teleportion_mat.shape
            #previous_pr = pr_score
            #pr_score = trans_mat.T.dot(pr_score)*alpha + (1-alpha)*np.ones((self.cluster_num,1))/float(self.cluster_num)
            pr_score = trans_mat.T.dot(pr_score)*alpha + (1-alpha)*teleportion_mat
            pr_score = normalize(pr_score,norm='l1',axis=0)
            print pr_score.shape, "shape"
            ite += 1
        self.pr_score = pr_score
        #self.page_pr_score = pr_score/
        self.avg_pr_score = pr_score.sum(axis=0)/self.cluster_num
        self.num_mat = num_mat
        self.output_pagerank()
        print self.avg_pr_score, "average score is "
        #self.cluster_num -= 1
    #
    def compute_indegree(self):
        #self.avg_indegree = self.indegree.sum(axis=0)/self.cluster_num
        print self.indegree, " indegree table"
        #for i,value in enumerate(self.indegree):
        #    print i, float(value)/self.counts_dict[i]
        #print self.avg_indegree, "average indegree is "


    def output_pagerank(self):
        print "page rank scores"
        for i in range(self.cluster_num):
            print "pagerank score for {0} is {1} and after average is {2}".format(i,self.pr_score[i][0], self.pr_score[i][0]/self.num_mat[i][0])

    def output_hits(self):
        print "authorative scores"
        for i in range(self.cluster_num):
            print "the authority score for {0} is {1}".format(i,self.auth_score[i])
        print "hub score "
        for i in range(self.cluster_num):
            print "the hub score for {0} is {1}".format(i,self.hub_score[i])

    # from dict to matrix
    def trans_dict_to_matrix(self):
        print self.cluster_num, "cluster number is "
        print self.target_cluster, "target cluster is "
        # +1 if consider outlier -1
        trans_mat = np.zeros((self.cluster_num,self.cluster_num))
        #for row,trans in self.trans_prob_mat.iteritems():
        print self.transition_mat, "transition matrix"
        for row,trans in self.adjacency_mat.iteritems():

            if int(row) == -1:
                #continue
                # use the last index for outlier
                row = self.cluster_num-1
            for col, value in trans.iteritems():
                #print col,value, "col and value"
                if col == -1:
                    #print col,value, "col and value"
                    #continue
                    col = self.cluster_num-1
                if self.rank_algo == "target":
                    if col == self.target_cluster:
                        trans_mat[row,col] = value
                else:
                    if col != row:
                        trans_mat[row,col] = value
            '''
            print row,type(row), trans, type(trans), "element of transition matrix"
            for col, value in trans.iteritems():
                if col == row:
                    trans_mat[row,col] = 0.0
                else:
                    trans_mat[row,col] = value
            '''
        # axis = 0 in-degree / axis = 1 out-degree
        print trans_mat.sum(axis=1), "out-degree"
        indegree_list = trans_mat.sum(axis=0)
        print indegree_list, "in-degree"
        self.indegree = indegree_list

        return trans_mat

    def analyze_navigation_table(self):
        #for key, trans in self.navigation_table.iteritems():
        #    print key, trans
        print len(self.navigation_table), "size of navigation table no zero "

    def get_ratio(self):
        counter = Counter(sitemap.UP_pages.category)
        total = sum(counter.values())
        self.cluster_ratio = defaultdict(float)
        for key in counter:
            self.cluster_ratio[key] = float(counter[key])/float(total)

    def get_threshold(self):
        score_list = []
        for i in range(self.cluster_num-1):
            tmp = (self.hub_score[i] + self.auth_score[i])
            tmp *= self.cluster_ratio[i]
            score_list.append(tmp)
        score_array = np.array(score_list)
        mean = np.mean(score_array)
        std = np.std(score_array)
        print mean, std
        threshold = mean - std
        print score_list
        print threshold

    def plot_distribution(self):
        x = []
        y = []
        print self.sitemap.intra_similarity
        for key in range(self.cluster_num-1):
            info_score = self.auth_score[key] + self.hub_score[key]
            print 'key is ',key
            Dsim = self.sitemap.intra_similarity[key]
            score = info_score
            size = self.cluster_ratio[key] * float(len(self.sitemap.path_list))
            print key, score, size
            x.append(score)
            y.append(size)
        plt.plot(x,y,"o",label="cluster point")
        plt.legend
        plt.xlabel('Score', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        print x
        print y
        plt.show()

    def load_bfs_corpus(self):
        self.corpus_set = set()
        file = "./results/bfs/{}_{}_0_bfs_size10001.txt".format(self.dataset,self.date)
        for line in open(file,"r").readlines():
            url = line.strip().split("\t")[0]
            self.corpus_set.add(url)

    def judge_in_corpus(self,url):
        return True
        if self.rank_algo == "bfs":
            return True
        else:
            if url in self.corpus_set:
                return True
            else:
                return False


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",help="The dataset to crawl")
    parser.add_argument("date",help="The date of sampling")
    #parser.add_argument('entry', help='The entry page')
    #parser.add_argument('prefix', help='For urls only have partial path')
    parser.add_argument('eps',help="the eps for dbscan")
    parser.add_argument('cluster',help="which cluster do we want to crawl? denoted by the rank of size, from index 0")
    parser.add_argument('crawl_size',help="crawling size")
    parser.add_argument('rank_algo',help="the method used to rerank crawling queue.")
    parser.add_argument('sitemap_size', default=None, help="the size of sitemap")
    args = parser.parse_args()


    # get entry and prefix
    if args.dataset == "stackexchange":
        args.entry, args.prefix =  "http://android.stackexchange.com/questions", "http://android.stackexchange.com"
    elif args.dataset == "asp":
        args.entry, args.prefix = "http://forums.asp.net/","http://forums.asp.net/"
    elif args.dataset == "youtube":
        args.entry, args.prefix =  "https://www.youtube.com/","https://www.youtube.com"
    elif args.dataset == "hupu":
        args.entry, args.prefix = "http://voice.hupu.com/hot","http://voice.hupu.com"
    elif args.dataset == "rottentomatoes":
        args.entry, args.prefix = "http://www.rottentomatoes.com","http://www.rottentomatoes.com"
    elif args.dataset == "douban":
        args.entry, args.prefix = "http://movie.douban.com","http://movie.douban.com"

    #elif args.dataset == "asp":
    if int(args.sitemap_size) == 1000:
        args.sitemap_size = None

    c = crawler(args.dataset,args.date,args.entry,args.prefix,float(args.eps),int(args.cluster),int(args.crawl_size),args.sitemap_size,args.rank_algo)
    pages = c.sitemap.UP_pages
    sitemap = c.sitemap

    c.analyze_navigation_table()


    sitemap.calculate_cluster_similarity()
    #sitemap.calculate_cluster_importance()
    c.compute_pagerank_scores()
    c.compute_indegree()
    c.compute_hit_scores()
    c.get_ratio()
    print c.navigation_table
    #c.plot_distribution()
    #c.get_threshold()
    c.load_bfs_corpus()
    if c.rank_algo == "est_pagerank" or c.rank_algo=="uniform" or c.rank_algo == "visit_ratio" or c.rank_algo=="pagerank" or c.rank_algo=="est_ratio":
        c.sampling(c.crawl_size,c.rank_algo)
    else:
        c.crawling(c.crawl_size)
    #counter = Counter(sitemap.UP_pages.category)
    #print counter

    #c.sampling(c.crawl_size,method=args.rank_algo)
    #print c.cluster_num
    #c.test_destination()

