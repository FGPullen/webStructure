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

class crawler:

    def __init__(self, dataset, date, entry,prefix, eps, cluster_rank,crawl_size, rank_algo="bfs"):
        self.dataset = dataset
        self.date = date
        self.eps = eps
        self.cluster_rank = cluster_rank
        self.rank_algo = rank_algo
        self.crawl_size = crawl_size
        self.rules = self.get_rules()
        self.entry, self.prefix = entry,prefix
        self.history_set = set()
        self.path_prefix = "../../Crawler/{}_samples/{}/".format(date,dataset.replace("new_",""))
        self.folder_path = ["../../Crawler/{}_samples/{}/".format(date,dataset.replace("new_",""))]
        self.sitemap = pageCluster(dataset,date,self.folder_path,0)
        self.full_folder = "../../Crawler/full_data/" + dataset
        self.trans = {}
        self.queue = {}
        self.crawled_cluster_count = defaultdict(int)
        self.trans_dict = read_trans_dict(dataset,date)


        #self.cluster_dict = get_cluster_dict(dataset,date)
        #self.cluster_num = int(self.sitemap.DBSCAN(eps_val=self.eps))
        self.cluster_num = int(self.sitemap.DBSCAN())
        self.build_gold_cluster_dict()
        self.cluster_xpath_trans = self.get_xpath_transition()
        self.trans_prob_mat = self.calculate_trans_prob_mat()
        self.max_score = 500
        self.target_cluster = self.get_sample_cluster()
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
        counts_dict = defaultdict(int)  # (cluster_id, xpath) -> int
        xpath_counts_dict = defaultdict(lambda : defaultdict(float)) # (cluster_id, xpath) - > dict[cluster_id] -> int

        trans_dict = read_trans_dict(self.dataset,self.date)  # [page][xpath] = [url list] ->[cluster][xpath] = {probability list}
        print "sample_url", sampled_urls
        for page_path, trans in trans_dict.iteritems():
            page_url = page_path.replace(".html","").replace("_","/")
            if page_url not in self.gold_dict:
                #print "?" + page_url, " is missing"
                continue
            else:
                cluster_id = self.cluster_dict[page_url]

            for xpath,url_list in trans.iteritems():
                length = len(url_list)
                count = 0
                for path in url_list:
                    url = path.replace(".html","").replace("_","/")
                    if url in sampled_urls:
                        count += 1

                #print "for xpath: {0} --- {1} out of {2} have been crawled and have cluster id".format(xpath,count, length)
                if count == 0:
                    continue
                else:
                    #if cluster_id == 1:
                    #    print page_path, xpath, url_list, "xpath_url_list in train"
                    key = (cluster_id,xpath)
                    #if key == (1,"/html/body/div/div/div/div/div/div/div/div/div/div/ul/li/div/div/div/div/div/div/div/ul/li/div/div/div/h3/a[yt-uix-sessionlink yt-uix-tile-link  spf-link  yt-ui-ellipsis yt-ui-ellipsis-2]"):
                    #    print page_path,url_list, "why 9 not 7???"
                    counts_dict[key] += 1
                    ratio = float(length)/float(count)
                    for path in url_list:
                        url = path.replace(".html","").replace("_","/")
                        if url in sampled_urls:
                            destination_id = self.cluster_dict[url]
                            #print url, destination_id
                            xpath_counts_dict[key][destination_id] += 1 * ratio
                    #if cluster_id == 1:
                    #    print ""

        # average
        for key,count in counts_dict.iteritems():
            for destination_id in xpath_counts_dict[key]:
                xpath_counts_dict[key][destination_id] /= count

        print "Micro average entropy is " + str(self.entropy(xpath_counts_dict))

        ''' output
        for key in xpath_counts_dict:
            if key[0] == 1:
                print key, xpath_counts_dict[key]
        '''
        print "=========== end of training ============"
        self.xpath_counts_dict = xpath_counts_dict
        return xpath_counts_dict
    # evaluate the entropy of xpath estination
    # micro output average for each xpath
    def entropy(self,xpath_counts_dict):
        entropy_list = []
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
            url = path.replace(self.path_prefix.replace("_","/"),"")
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
                    #if self.match_rules(link,[["members"]]):
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
        else:
            return 0

    def get_sample_cluster(self):
        if self.dataset=="youtube":
            cluster_list = ["../../Crawler/May1_samples/youtube/https:__www.youtube.com_watch?v=zhnMSVb0oYA.html",\
                            "../../Crawler/May1_samples/youtube/https:__www.youtube.com_playlist?list=PLuKg-WhduhkmIcFMN7wxfVWYu8qnk0jMN.html",\
                            "../../Crawler/May1_samples/youtube/https:__www.youtube.com_user_vat19com_discussion.html"]
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
            cluster_list = ["../../Crawler/May1_samples/douban/https:__movie.douban.com_review_7773468_.html",\
                            "../../Crawler/May1_samples/douban/https:__movie.douban.com_subject_10727641_questions_ask_?from=subject.html"]
            file_path = cluster_list[self.cluster_rank]
        page,cluster_id = self.classify(file_path)
        self.test_page = file_path
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


        write_file = open("./results/sampling/random_{0}_{1}_size{2}.txt".format(method,self.dataset,self.crawl_size),"w")
        num_web_crawl=0
        entry, prefix = self.entry, self.prefix
        self.url_stack = [entry]
        self.final_list, url_list, last_list = [], [], []
        size, num = num_crawl,  0 # the number of crawling
        s = sampler(self.dataset,self.entry,self.prefix,0)
        while(num<size and len(self.url_stack) >0):
            first_url = self.url_stack[0]
            print "first_url", first_url
            try:
                sys.stdout.write("num is {}\n".format(num))
                sys.stdout.flush()
                #print num, "num"
                url_list,cluster_id = self.sample_link(first_url,s,method)
                if first_url not in self.history_set:
                    self.final_list.append((first_url,cluster_id))
                    num += 1
            except:
                print "might miss somthing here"
                traceback.print_exc()
                flag = s.crawlUrl(first_url,self.dataset,self.url_stack,self.history_set)
                if flag == 1:
                    sys.stdout.write("num is {}\n".format(num))
                    sys.stdout.flush()
                    #print num, "num"
                    url_list,cluster_id = self.sample_link(first_url,s,method)
                    print url_list
                    if first_url not in self.history_set:
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
                self.history_set.add(first_url)

            probability = 0.15
            if method=="uniform":
                # after processing, 0.15 random and 0.85 uniform sampling
                # no out-links - random sample
                if random.random() < probability:
                    self.select_from_history_set()
                else:
                    try:
                        print url_list
                        id = random.randrange(len(url_list))
                        self.url_stack.append(url_list[id])
                        print url_list[id], "select from out-links"
                    except:
                        self.select_from_history_set()

            elif method == "pagerank":
                if random.random() < probability:
                    url = random.sample(self.history_set,1)[0]
                    self.url_stack.append(url)
                    print url, "random sampled from history set"
                else:
                    try:
                        id = self.sample_from_dist(url_list,pr_dict,avg_pr)
                        self.url_stack.append(url_list[id])
                        print url_list[id], "select from out-links"
                    except:
                        self.select_from_history_set()
            elif method == "indegree":
                print "sample from orcacle indegree"
                if random.random() < probability:
                    url = random.sample(self.history_set,1)[0]
                    self.url_stack.append(url)
                    print url, "random sampled from history set"
                else:
                    try:
                        id = self.sample_from_dist(url_list,indegree_dict,avg_indegree)
                        self.url_stack.append(url_list[id])
                        print url_list[id], "select from out-links"
                    except:
                        traceback.print_exc()
                        self.select_from_history_set()

            else: # our method
                if random.random() < probability:
                    url = random.sample(self.history_set,1)[0]
                    self.url_stack.append(url)
                    print url, "random sampled from history set"
                else:
                    try:
                        id = self.sample_from_prob_list(url_list)
                        self.url_stack.append(url_list[id][0])
                        print url_list[id], "select from out-links"
                    except:
                        self.select_from_history_set()


        print len(self.final_list), "length of final list"
        for pair in self.final_list:
            url, cluster_id = pair[0],pair[1]
            write_file.write(url + "\t"+ str(cluster_id) + '\n')
                #random sample one from url list and add to ur_stack

    def select_from_history_set(self):
        url = random.sample(self.history_set,1)[0]
        self.url_stack.append(url)
        print url, "random sampled from history set"


    # input: entry, prefix, site, trans_xpath_dict , target_cluster id
    # target: find the cluster page that we want
    def crawling(self,num_crawl):

        # self.entry, self.prefix, self.dataset, self.trans_xpath_dict, target_cluste id
        #self.target_cluster = self.get_sample_cluster()
        write_file = open("./results/{0}_{1}_{2}_{3}_size{4}.txt".format(self.dataset,self.date, self.cluster_rank, self.rank_algo, self.crawl_size),"w")
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
            print self.url_stack[0]
            print self.url_stack[-1]

            #first_url = self.url_stack[0][0]
            try:
                print "first url is " + first_url

            except:
                traceback.print_exc()
            if first_url not in self.history_set:
                num += 1
                try:
                    url_list,cluster_id = self.crawl_link(first_url,  self.history_set, s)
                    #print "url_list", url_list
                    self.sort_queue(url_list,first_url,self.rank_algo) # sort url_stack
                    self.final_list.append((first_url,parent_url,parent_xpath,score,cluster_id))
                except:
                    print "might miss somthing here"
                    traceback.print_exc()
                    flag = s.crawlUrl(first_url,self.dataset,self.url_stack,self.history_set)
                    if flag == 1:
                        url_list,cluster_id = self.crawl_link(first_url,  self.history_set,s )
                        self.sort_queue(url_list,first_url,rank_algo=self.rank_algo)
                        self.final_list.append((first_url,parent_url,parent_xpath,score,cluster_id))
                        random_time_s = random.randint(5, 10)
                        time.sleep(random_time_s)
                        num_web_crawl += 1
                        if num_web_crawl%10 == 9:
                            random_time_s = random.randint(60, 120)
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
            if url[0] not in self.history_set and url not in self.url_stack:
                self.url_stack += url_list

        self.url_stack.pop(0) # delete current page
        if rank_algo == "bfs":
            print "bfs for crawling frontier"
            #time.sleep(10)
            print self.url_stack[0], "first element in url stack"
        elif rank_algo == "sort":
            print "algo is sorting by hub and authority"
            self.url_stack = sorted(self.url_stack,key=lambda tup:tup[-1],reverse=True)  # sorts in place
            #print "===" + str(self.url_stack) + "==="
            '''
            for url in self.url_stack:
                url = url[0]
                if self.match_rules(url,self.rules):
                    print url,
            print ""
            #time.sleep(2)
            print self.url_stack[0], "first element in url stack"
            '''
        elif rank_algo == "general":
            print "general crawling which takes the ratio of clusters into consideration"

        else:
            raise



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
        print link_dict, "link dict"
        #self.transition_dict[url] = link_dict
        for xpath in link_dict:
            # considering cluster
            #distribution = self.cluster_xpath_trans[(cluster_id,xpath)]
            distribution = self.xpath_counts_dict[(cluster_id,xpath)]
            #print distribution, (cluster_id,xpath), " cluster_id, xpath for distribution"
            score = self.calculate_url_importance(distribution) # and self.cluster_trans_prob_mat
            #print distribution, "the probability of itself"
            link_list = link_dict[xpath]
            flag = 0
            for url in link_list:
                if flag == 0:
                    #print url ,score, "url , score"
                    flag += 1

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
        if method == "uniform" or method == "pagerank" or method == "indegree":
            print "no predicting scores"
            for xpath in link_dict:
                link_list = link_dict[xpath]
                for link in link_list:
                    if link not in self.history_set:
                        available_url_list.append(link)
        else:
            print "predicting scores based on distribution"
            for xpath in link_dict:
                distribution = self.xpath_counts_dict[(cluster_id,xpath)]
                #print distribution, " distribution is "
                link_list = link_dict[xpath]
                if method == "est_pagerank":
                    score = self.calculate_url_pr(distribution)
                elif method == "est_degree":
                    score = self.calculate_url_indegree(distribution)
                #print score, "calcualte_url_pr"
                for link in link_list:
                    if link not in self.history_set:
                        available_url_list.append((link,score))
        #print "sample link ends"
        return available_url_list, cluster_id

    # input: url_list and its pagerank score distribution dict (key->value)
    # output: an id sampled inversed to its pagerank score
    def sample_from_dist(self,url_list,pr_dict,avg_pr):
        pr_dist = []
        #print url_list
        for url in url_list:
            if url in pr_dict:
                if pr_dict[url] !=0:
                    pr_dist.append(1.0/pr_dict[url])
                else:
                    pr_dist.append(1.0/float(avg_pr))
            else:
                pr_dist.append(1.0/float(avg_pr))
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
        return random_ind


    # input, list of pairs (url,score)
    def sample_from_prob_list(self,url_list):
        pr_dist = []
        for item in url_list:
            pr_dist.append(1.0/item[-1])
        print pr_dist, " reversed prob list"
        pr_dist = normalize(pr_dist,"l1")[0]
        #print url_list, " estimated pagerank prob distribution"
        #print pr_dist, " reversed probability distribution"
        if len(pr_dist) == 0:
            raise Exception("Has no links")

        cdf = [pr_dist[0]]
        for i in xrange(1, len(pr_dist)):
            cdf.append(cdf[-1] + pr_dist[i])
        random_ind = bisect(cdf,random.random())
        print random_ind, "random sampling id from pagerank score "
        return random_ind


    # @ input: cluster_xpath_trans (cluster,xpath) -> distribution
    # @ output:  self.trans_prob_mat, [cluster_id][cluster_id] -> n
    def calculate_trans_prob_mat(self):
        trans_prob_mat = defaultdict(lambda : defaultdict(float))
        for pair, distribution in self.cluster_xpath_trans.iteritems():
            start_id, xpath = pair[0], pair[1]
            for dest_id,num in distribution.iteritems():
                trans_prob_mat[start_id][dest_id] += num

        print trans_prob_mat
        #self.cluster_num = max(trans_prob_mat.keys())+1
        return trans_prob_mat

    def calculate_url_importance(self,distribution,target_weight=None):
        k1, k2, k3 = 0.4, 0.1, 0.5
        score = 0.0
        total = sum(self.crawled_cluster_count.values())
        norm = sum(distribution.values())
        for dest_id, num in distribution.iteritems():
            prob = float(num)/float(norm)
            #discount = (1-float(self.crawled_cluster_count[dest_id])/float(total))
            ratio = float(self.crawled_cluster_count[dest_id])/float(total)
            ratio_weight = 1 - ratio**2
            #print total, "total", discount, "discount"
            score += k1* prob * self.auth_score[dest_id]
            score += k2* prob * self.hub_score[dest_id]
            score += k3 * prob * ratio_weight
        return score

    def calculate_url_pr(self,distribution):
        #print distribution, " distribution "
        if len(distribution) == 0:
            return self.avg_pr_score

        score, norm = 0.0, sum(distribution.values())
        for dest_id, num in distribution.iteritems():
            prob = float(num)/float(norm)
            score += prob * self.pr_score[dest_id][0]
            #print self.pr_score[dest_id][0]
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


    def compute_hit_scores(self,max_iter=100):
        trans_mat = self.trans_dict_to_matrix() # need normalization
        auth_score = np.ones((self.cluster_num, 1)) / float(self.cluster_num)
        hub_score = np.ones((self.cluster_num, 1)) / float(self.cluster_num)
        # previous_pr = np.zeros((doc_num,1))
        ite = 0
        while (ite < max_iter):
            # previous_pr = pr_score
            hub_score = trans_mat.dot(auth_score)
            auth_score = trans_mat.T.dot(hub_score)
            auth_score = normalize(auth_score, norm='l2', axis=0)
            hub_score = normalize(hub_score, norm='l2', axis=0)
            ite += 1
        self.auth_score = auth_score
        self.hub_score = hub_score
        self.output_hits()  # n*1

    def compute_pagerank_scores(self,max_iter=100):
        alpha = 0.15
        trans_mat = self.trans_dict_to_matrix()
        trans_mat = normalize(trans_mat, norm='l1', axis=1)
        pr_score = np.ones((self.cluster_num,1))/float(self.cluster_num)
        ite = 0
        while(ite < max_iter):
            print "iteration ", ite
            #previous_pr = pr_score
            pr_score = trans_mat.T.dot(pr_score)*alpha + (1-alpha)*np.ones((self.cluster_num,1))/float(self.cluster_num)
            ite += 1
        self.pr_score = pr_score
        self.avg_pr_score = pr_score.sum(axis=0)/self.cluster_num
        self.output_pagerank()
        print self.avg_pr_score, "average score is "

    #
    def compute_indegree(self):
        self.avg_indegree = self.indegree.sum(axis=0)/self.cluster_num
        print self.indegree, " indegree table"
        print self.avg_indegree, "average indegree is "


    def output_pagerank(self):
        print "page rank scores"
        for i in range(self.cluster_num):
            print "pagerank score for {0} is {1}".format(i,self.pr_score[i][0])

    def output_hits(self):
        print "authorative scores"
        for i in range(self.cluster_num):
            print "the authority score for {0} is {1}".format(i,self.auth_score[i][0])
        print "hub score "
        for i in range(self.cluster_num):
            print "the hub score for {0} is {1}".format(i,self.hub_score[i][0])

    def trans_dict_to_matrix(self):
        print self.cluster_num, "cluster number is "
        print self.target_cluster, "target cluster is "
        trans_mat = np.zeros((self.cluster_num,self.cluster_num))
        for row,trans in self.trans_prob_mat.iteritems():
            if int(row) == -1:
                continue
            for col, value in trans.iteritems():
                if self.rank_algo == "sort":
                    if col == self.target_cluster:
                        trans_mat[row,col] = value
                else:
                    trans_mat[row,col] = value
        # axis = 0 in-degree / axis = 1 out-degree
        print trans_mat.sum(axis=1), "out-degree"
        indegree_list = trans_mat.sum(axis=0)
        print indegree_list, "in-degree"
        self.indegree = indegree_list

        return trans_mat

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",help="The dataset to crawl")
    parser.add_argument("date",help="The date of sampling")
    parser.add_argument('entry', help='The entry page')
    parser.add_argument('prefix', help='For urls only have partial path')
    parser.add_argument('eps',help="the eps for dbscan")
    parser.add_argument('cluster',help="which cluster do we want to crawl? denoted by the rank of size, from index 0")
    parser.add_argument('crawl_size',help="crawling size")
    parser.add_argument('rank_algo',help="the method used to rerank crawling queue.")
    args = parser.parse_args()

    c = crawler(args.dataset,args.date,args.entry,args.prefix,float(args.eps),int(args.cluster),int(args.crawl_size),args.rank_algo)
    pages = c.sitemap.UP_pages
    sitemap = c.sitemap
    c.compute_hit_scores()
    c.compute_pagerank_scores()
    c.compute_indegree()
    #c.crawling(c.crawl_size)
    c.sampling(c.crawl_size,method=args.rank_algo)
    #print c.cluster_num
    #c.test_destination()

