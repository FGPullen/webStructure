
import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from hits_estimate import read_trans_dict
import math
from collections import defaultdict,Counter
import re
import traceback
import random,time
import numpy as np
from sample import sampler
import matplotlib.pyplot as plt





class eval_crawl:

    def __init__(self,dataset, date,cluster_rank, crawl_size ,rank_algo, test_size):
        self.dataset = dataset
        self.date = date
        self.cluster_rank = cluster_rank
        self.crawl_size = crawl_size
        self.rank_algo = rank_algo
        self.test_size = test_size
        self.match_list = []
        self.relevant_list = []
        self.get_match_list(test_size)
        self.write_file = open("target_cluster.results","a")

    def get_rules(self):
        if self.dataset=="stackexchange":
            cluster_list = [[["a","^[0-9]+$"],["q","^[0-9]+$"],["questions","^[0-9]+$"]],\
                            [["users","^[0-9]+(.*)$"]],[["questions","tagged"],["^questions?(.*)=(.*)$"],["unanswered","tagged"]]]
            rules = cluster_list[self.cluster_rank]
        elif self.dataset == "youtube":
            cluster_list =[[["^watch\?v=(.*)$"]],[["channel","^(.*)$"],["user","^playlists(.*)$"],["user","^videos(.*)$"],["user","^discussion(.*)$"],["user","^(.*)$"]],[["^playlist\?(.*)$"]]]
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
                            [["subject","^[0-9]+$","^(.*)from(.*)$"],["subject","^[0-9]+$"]]]
            rules = cluster_list[self.cluster_rank]
        elif self.dataset == "hupu":
            cluster_list = [[["cba","^[0-9]+(.*)$"],["china","^[0-9]+(.*)$"],["f1","^[0-9]+(.*)$"],["other","^[0-9]+(.*)$"],["soccer","^[0-9]+(.*)$"],["sports","^[0-9]+(.*)$"],\
              ["tennis","^[0-9]+(.*)$"],["zb","^[0-9]+(.*)$"],["nba","^[0-9]+(.*)$"],["wcba","^[0-9]+(.*)$"]],[["o"],["people","^[0-9]+(.*)$"]]]
            rules = cluster_list[self.cluster_rank]
        elif self.dataset == "rottentomatoes":
            cluster_list = [[["m","^(.*)$"],["tv","^(.*)$"]],\
                            [["celebrity"]]]
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

    def get_match_list(self,test_size):
        rules = self.get_rules()
        url_list, match_list, hub_list, relevant_list, hub_set = [], [], [], [], set()
        if self.rank_algo == "bfs":
            file_path = "./results/bfs/{0}_{1}_0_{3}_size{4}.txt".format(self.dataset, self.date, self.cluster_rank, self.rank_algo,self.crawl_size)
        elif self.rank_algo == "TPM":
            file_path = "./results/evaluate/baselines/vidal_{0}_{1}_{2}_size{4}.txt".format(self.dataset, self.date, self.cluster_rank, self.rank_algo,self.crawl_size)
        elif "sitemap" in self.rank_algo:
            #try:
            #file_path = "./results/evaluate/sitemap/{0}_{1}_{2}_target_{3}_size1001.txt".format(self.dataset, self.date, self.cluster_rank, self.rank_algo)
            #except:
            file_path = "./results/evaluate/sitemap/{0}_{1}_{2}_target_{3}_size{4}.txt".format(self.dataset, self.date, self.cluster_rank, self.rank_algo,self.crawl_size)
        else:
            file_path = "./results/evaluate/target/{0}_{1}_{2}_{3}_size5001.txt".format(self.dataset, self.date, self.cluster_rank, self.rank_algo,self.crawl_size)
        print file_path , 'file path'
        result_lines = open(file_path,"r").readlines()

        for index,line in enumerate(result_lines):
            if index >= test_size:
                break
            if len(line.strip().split("\t")) ==3:
                url, parent_url = line.split("\t")[0],line.split("\t")[1]
            elif len(line.strip().split("\t")) != 5:
                url = line.strip()
                parent_url = ""
            else:
                [url, parent_url, parent_xpath,score,cluster] = line.strip().split("\t")
            if self.match_rules(url,rules):
                flag = 1
                hub_set.add(parent_url)
                url_list.append(url)
            else:
                flag = 0
            match_list.append(flag)

        print len(match_list), len(url_list), "length of all list that match pattern"

        self.match_list = match_list
        self.url_list = url_list
        self.rules = rules

    def eval(self,crawl_num,mode="match"):
        if mode=="match":
            print " in match", crawl_num,
            self.match_list = self.match_list[1:crawl_num]
            right = sum(self.match_list[1:crawl_num])
            total = len(self.match_list[1:crawl_num])
            print right, total, "right and total"
            precision = float(right)/float(total)
        elif mode == "relevant":
            right = sum(self.relevant_list[1:crawl_num])
            total = len(self.relevant_list[1:crawl_num])
            precision = float(right)/float(total)
        print "precision is ", precision
        self.write_file.write("{0} {1} {2} precision {3}".format(self.dataset,self.rank_algo,self.cluster_rank,precision)+"\n")
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
                    #print rule[match_id]
            else:
                if term == rule[match_id]:
                    match_id += 1
            if match_id >= len(rule):
                break

        if match_id >= len(rule):
            return True
        else:
            return False


def adding_to_recall_set(v,recall_set,num_crawl):
    num_crawl = min(len(v.match_list),num_crawl)
    number = 0
    print "num_crawl", num_crawl
    for i in range(1,num_crawl):
        if v.match_list[i]==1:
            number += i
    print "the number of match!", number
    url_sub_list = v.url_list[:number]
    for url in url_sub_list:
        recall_set.add(url)

    print len(recall_set), "recall _set"


def calculate_recall(v,recall_set,num_crawl):
    #assert len(l1) == len(l2)
    num_crawl = min(len(v.match_list),len(v.url_list),num_crawl)
    print num_crawl ,"num crawl"
    total, length = len(recall_set), 0
    url_list = v.url_list[1:num_crawl]
    for url in url_list:
        if url in recall_set:
            length +=1
    recall = length/float(total)
    print length,total, "recall is {}".format(recall)
    v.write_file.write("{0} {1} {2} recall {3}".format(v.dataset,v.rank_algo,v.cluster_rank,recall)+"\n")
    return float(length)/float(total)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",help="The dataset to crawl")
    parser.add_argument("date",help="The date of sampling")
    parser.add_argument('cluster',help="which cluster do we want to crawl? denoted by the rank of size, from index 0")
    parser.add_argument('crawl_size',help="crawling size")
    parser.add_argument('rank_algo',help="rank algo")



    for cluster_id in [0,1]:
        p_1000 = []
        p_500 = []
        p_200 = []
        p_100 = []
        for dataset in ["asp","youtube","douban","hupu","stackexchange"]:
            crawl_size = 2000
            args = parser.parse_args()
            our = eval_crawl(dataset,args.date,cluster_id,int(args.crawl_size),args.rank_algo,crawl_size)
            #bfs = eval_crawl(dataset,args.date,int(args.cluster),10001,"bfs",crawl_size)
            #tpm = eval_crawl(dataset,args.date,int(args.cluster),1001,"TPM",crawl_size)
            sitemap100 = eval_crawl(dataset,args.date,cluster_id,2000,"sitemap100",crawl_size)
            sitemap200 = eval_crawl(dataset,args.date,cluster_id,2000,"sitemap200",crawl_size)
            sitemap500 = eval_crawl(dataset,args.date,cluster_id,2000,"sitemap500",crawl_size)
            crawl_num = 3000
            p_1000.append(our.eval(crawl_num,mode="match"))
            #bfs.eval(crawl_num,mode="match")
            #tpm.eval(crawl_num,mode="match")
            p_500.append(sitemap500.eval(crawl_num,mode="match"))
            p_200.append(sitemap200.eval(crawl_num,mode="match"))
            p_100.append(sitemap100.eval(crawl_num,mode="match"))


        a_1000= np.mean(np.array(p_1000))
        a_500 =np.mean(np.array(p_500))
        a_200 =np.mean(np.array(p_200))
        a_100 =np.mean(np.array(p_100))

        print a_1000,a_500,a_200,a_100, "average "

        '''
        methods = [our,sitemap100,sitemap200]
        recall = set()
        for m in methods:
            for url in m.url_list:
                recall.add(url)
        print len(recall)

        '''
        print "p_1000", p_1000,np.mean(np.array(p_1000))
        print "p_500", p_500,np.mean(np.array(p_500))
        print "p_200", p_200,np.mean(np.array(p_200))
        print "p_100", p_100,np.mean(np.array(p_100))

        plt.plot([1000,500,200,100],[a_1000,a_500,a_200,a_100],label="cluster {}".format(cluster_id))
    plt.xlabel("Sitemap Size")
    plt.ylabel("Precision")
    plt.legend(loc=4)
    plt.show()
    #for url in recall:
    #    print "test", url

    #for m in methods:
    #    calculate_recall(m,recall,1001)




    '''
    recall = calculate_recall(target_match_url_list,bfs_match_url_list)
    print 'recall is ', recall

    total = len(bfs_match_url_list)
    max_recall = -1.0
    for rule in eval.rules:
        print rule
        num = 0
        for url in bfs_match_url_list:
            if eval.match(url,rule):
                print url,rule
                num += 1
        if num > max_recall:
            max_recall = num
    print "TPM recall is {}".format(float(max_recall)/float(total))

    '''
