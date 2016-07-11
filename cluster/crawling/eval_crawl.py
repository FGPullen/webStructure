
import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from hits_estimate import read_trans_dict
import math
from collections import defaultdict,Counter
import re
import traceback
import random,time
from sample import sampler


class eval_crawl:

    def __init__(self,dataset, date,cluster_rank, crawl_size ,rank_algo):
        self.dataset = dataset
        self.date = date
        self.cluster_rank = cluster_rank
        self.crawl_size = crawl_size
        self.rank_algo = rank_algo
        self.match_list = []
        self.relevant_list = []
        self.get_match_list()

    def get_rules(self):
        if self.dataset=="stackexchange":
            cluster_list = [[["a","^[0-9]+$"],["q","^[0-9]+$"],["questions","^[0-9]+$"]],\
                            [["users","^[0-9]+(.*)$"]]]
            rules = cluster_list[self.cluster_rank]
        elif self.dataset == "youtube":
            cluster_list =[[["^watch\?v=(.*)$"]],[["^playlist\?(.*)$"]]]
            rules = cluster_list[self.cluster_rank]
        elif self.dataset == "asp":
            cluster_list =[[["p","^[0-9]+$"],["post","^[0-9]+.aspx$"],["t","^[0-9]+.aspx(.*)$"],["t","next","^[0-9]+$"],["t","prev","^[0-9]+$"]],\
            [["^members(.*)$"]],[["login","^RedirectToLogin(.*)$"]]]
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

    '''
    def eval(self, crawl_num):
        rules = self.get_rules()
        if self.rank_algo == "bfs":
            file_path = "./results/bfs/{0}_{1}_0_{3}_size{4}.txt".format(self.dataset, self.date, self.cluster_rank, self.rank_algo,self.crawl_size)
        else:
            file_path = "./results/{0}_{1}_{2}_{3}_size{4}.txt".format(self.dataset, self.date, self.cluster_rank, self.rank_algo,self.crawl_size)
        #print file_path
        result_lines = open(file_path,"r").readlines()
        num,total,num_flag = 0, len(result_lines),0
        for line in result_lines:
            url = line.strip()
            num_flag += 1
            if num_flag > crawl_num:
                break
            if self.match_rules(url,rules):
                num+=1
        precision = float(num)/float(crawl_num)
        print precision
        return  precision
    '''

    def get_match_list(self):
        rules = self.get_rules()
        match_list, hub_list, relevant_list, hub_set = [], [], [], set()
        if self.rank_algo == "bfs":
            file_path = "./results/bfs/{0}_{1}_0_{3}_size{4}.txt".format(self.dataset, self.date, self.cluster_rank, self.rank_algo,self.crawl_size)
        else:
            file_path = "./results/{0}_{1}_{2}_{3}_size{4}.txt".format(self.dataset, self.date, self.cluster_rank, self.rank_algo,self.crawl_size)
        result_lines = open(file_path,"r").readlines()
        for line in result_lines:
            if len(line.strip().split("\t")) != 3:
                url = line.strip()
                parent_url = ""
            else:
                [url, parent_url, parent_xpath] = line.strip().split("\t")
            if self.match_rules(url,rules):
                flag = 1
                hub_set.add(parent_url)
            else:
                flag = 0
            match_list.append(flag)

        print hub_set, "hub url set"
        for line in result_lines:
            if len(line.strip().split("\t")) != 3:
                hub_list.append(0)
            else:
                [url, parent_url, parent_xpath] = line.strip().split("\t")
                if url in hub_set:
                    flag = 1
                else:
                    flag = 0
                hub_list.append(flag)
        print len(match_list), len(hub_list), len(result_lines), "length of all list"

        for i in range(len(match_list)):
            if match_list[i] or hub_list[i]:
                relevant_list.append(1)
            else:
                relevant_list.append(0)
        self.relevant_list = relevant_list
        self.match_list = match_list
        #print relevant_list
        #print match_list

    def eval(self,crawl_num,mode="match"):
        if mode=="match":
            right = sum(self.match_list[1:crawl_num])
            total = len(self.match_list[1:crawl_num])
            precision = float(right)/float(total)
        elif mode == "relevant":
            right = sum(self.relevant_list[1:crawl_num])
            total = len(self.relevant_list[1:crawl_num])
            precision = float(right)/float(total)
        #print precision
        return precision


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

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",help="The dataset to crawl")
    parser.add_argument("date",help="The date of sampling")
    parser.add_argument('cluster',help="which cluster do we want to crawl? denoted by the rank of size, from index 0")
    parser.add_argument('crawl_size',help="crawling size")
    parser.add_argument('rank_algo',help="rank algo")
    args = parser.parse_args()
    eval = eval_crawl(args.dataset,args.date,int(args.cluster),int(args.crawl_size),args.rank_algo)
    for crawl_num in range(50,350,50):
        print crawl_num
        eval.eval(crawl_num,mode="match")



