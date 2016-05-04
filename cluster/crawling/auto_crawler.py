import os.path
import sys

from hits_estimate import read_trans_dict

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from pageCluster import pageCluster
from page import Page
import math
from collections import defaultdict,Counter
import re

class crawler:

    def __init__(self, dataset, date, entry,prefix):
        self.dataset = dataset
        self.date = date
        self.path_prefix = "../../Crawler/{}_samples/{}/".format(date,dataset.replace("new_",""))
        self.folder_path = ["../../Crawler/{}_samples/{}/".format(date,dataset.replace("new_",""))]
        self.sitemap = pageCluster(dataset,date,self.folder_path,0)
        self.trans = {}
        self.queue = {}
        self.trans_dict = read_trans_dict(dataset,date)
        #self.cluster_dict = get_cluster_dict(dataset,date)
        self.sitemap.DBSCAN(eps_val=0.15)
        self.build_gold_cluster_dict()
        self.cluster_xpath_trans = self.get_xpath_transition()
        print self.cluster_xpath_trans, "self.cluster_xpath_trans"
        #raise

    # @ input: page object and self.sitemap
    # @ return: cluster id
    def classify(self,file_path):
        #self.sitemap
        page = Page(file_path)
        x = []
        for feat in self.sitemap.features:
            if feat in page.xpaths:
                # remember log !
                x.append(math.log(float(page.xpaths[feat]+1),2) * pages.idf[feat])
            else:
                x.append(0.0)
        pred_y = sitemap.nbrs.predict(x)[0]
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
                print "?" + page_url, " is missing"
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
        #raise

        ''' output
        for key in xpath_counts_dict:
            if key[0] == 1:
                print key, xpath_counts_dict[key]
        '''
        print "=========== end of training ============"

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
    def predict_destination(self,file_path,target_cluster):
        #print self.cluster_xpath_trans
        page,cluster_id = self.classify(file_path)

        if self.date == "May1":
            link_dict = page.getAnchor(True)
        else:
            link_dict = page.getAnchor()
        #print len(link_dict), "link_dict length"
        #print cluster_id, " cluster id"
        right,guess = 0,0

        for xpath in link_dict:
            #print file_path
            #print cluster_id,xpath, "xpath"
            #print link_dict[xpath], "links "
            #print self.cluster_xpath_trans[(cluster_id,xpath)], "distribution"
            watch_list = []
            #print self.cluster_xpath_trans[(cluster_id,xpath)]
            if target_cluster in  self.cluster_xpath_trans[(cluster_id,xpath)]:
                tmp = self.cluster_xpath_trans[(cluster_id,xpath)]
                print tmp, (cluster_id,xpath)
                # we only focus on intra links
                link_list = []
                for link in link_dict[xpath]:
                    if self.intraJudge(link,self.dataset)>0:
                        link_list.append(link)
                guess += len(link_list) * (tmp[target_cluster]/sum(tmp.values()))

                print xpath,(tmp[target_cluster]/sum(tmp.values())), link_list
                for link in link_list:
                    #if "playlist?" in link:
                    #if self.match_rules(link,[["a","^[0-9]+$"],["q","^[0-9]+$"],["questions","^[0-9]+$"]]):
                    #if self.match_rules(link,[["users","^[0-9]+(.*)$"]]):
                    #if self.match_rules(link,[["p","^[0-9]+$"],["post","^[0-9]+.aspx$"],["t","^[0-9]+.aspx(.*)$"],["t","next","^[0-9]+$"],["t","prev","^[0-9]+$"]]):
                    #if self.match_rules(link,[["members"]]):
                    #if self.match_rules(link,[["^Hotel_Review-(.*)$"]]):
                    #if self.match_rules(link,[["subject","^[0-9]+$"]]) and "" in link.split("/"):
                    if self.match_rules(link,[["subject","^[0-9]+$","questions","ask"]]):
                        watch_list.append(link)
                        right += 1
                print link_list
                print watch_list

        if right !=0:
            print right,guess
            precision = 1 - math.fabs(right-guess)/(right+guess)
            print link_dict
        else:
            precision = None
        print  precision, " 1-|right - guess|/right"
        return cluster_id,precision


    def intraJudge(self,url, site):
    # oulink with http or symbol like # and /
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
            #file_path = "../../Crawler/May1_samples/youtube/https:__www.youtube.com_watch?v=zhnMSVb0oYA.html"
            file_path = "../../Crawler/May1_samples/youtube/https:__www.youtube.com_playlist?list=PLuKg-WhduhkmIcFMN7wxfVWYu8qnk0jMN.html"
        elif self.dataset == "stackexchange":
            #file_path = "../../Crawler/May1_samples/stackexchange/http:__android.stackexchange.com_q_138084.html"
            file_path = "../../Crawler/May1_samples/stackexchange/http:__android.stackexchange.com_users_134699_rood.html"
        elif self.dataset == "asp":
            #file_path = "../../Crawler/May1_samples/asp/http:__forums.asp.net__post_6027404.aspx.html"
            file_path = "../../Crawler/May1_samples/asp/http:__forums.asp.net__members_sapator.aspx.html"
        elif self.dataset == "tripadvisor":
            file_path = "../../Crawler/May1_samples/tripadvisor/http:__www.tripadvisor.com_Hotel_Review-g48016-d655840-Reviews-Park_Lane_Motel-Lake_George_New_York.html.html"
        elif self.dataset == "douban":
            #file_path = "../../Crawler/May1_samples/douban/https:__movie.douban.com_subject_10757577_.html"
            file_path = "../../Crawler/May1_samples/douban/https:__movie.douban.com_subject_10727641_questions_ask_?from=subject.html"
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

    # input: entry, prefix, site, trans_xpath_dict , target_cluster id
    # target: find the cluster page that we want
    '''
    def crawling(self,link):
        parse the file and extract link
        get destination url
        classify page
        use self.trans to maintain self.queue

    '''



    '''
    # @ input transition matrix file
    # @ output two diction auth and hub: key: cluster_id , value: score
    def calculate_hits(self):


    def sort_queue(self):

    '''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",help="The dataset to crawl")
    parser.add_argument("date",help="The date of sampling")
    parser.add_argument('entry', help='The entry page')
    parser.add_argument('prefix', help='For urls only have partial path')
    args = parser.parse_args()
    c = crawler( args.dataset,args.date,args.entry,args.prefix)
    pages = c.sitemap.UP_pages
    sitemap = c.sitemap
    target_cluster = c.get_sample_cluster() #
    folder_path = "../../Crawler/Mar15_samples/{}/".format(args.dataset)
    macro_average, num = 0, 0
    test_cluster_list = []

    for file in os.listdir(folder_path):
        file_path = folder_path + file
        print file_path, "for test set Mar 15"
        cluster_id, precision = c.predict_destination(file_path,target_cluster)
        test_cluster_list.append(cluster_id)
        if precision is not None:
            macro_average += precision
            num += 1
    print float(macro_average)/float(num)
    c = Counter(test_cluster_list)
    print c, "counter"











    '''
    for index, page in enumerate(pages.pages):
        print page.path, pages.category[index]


    folder_path = "../../Crawler/Mar15_samples/{}/".format(args.datasets.replace("new_",""))
    gold_file = open("../Annotation/site.gold/{0}/{0}.gold".format(args.datasets.replace("new_","")),"r").readlines()
    gold_dict = {}
    for line in gold_file:
        [key,label] = line.strip().split("\t")
        gold_dict[key] = label

    right = 0
    total = 0

    for file in os.listdir(folder_path):
        total += 1
        label = gold_dict[file]
        file_path = folder_path + file
        print file_path
        page = Page(file_path)
        x = []
        for feat in sitemap.features:
            if feat in page.xpaths:
                # remember log !
                x.append(math.log(float(page.xpaths[feat]+1),2) * pages.idf[feat])
            else:
                x.append(0.0)
        pred_y = sitemap.nbrs.predict(x)[0]
        if pred_y == int(label):
            right += 1
            dist,index = sitemap.nbrs.kneighbors(x)
            #print index[0]
            #for id in index[0]:
            #    print pages.pages[id].path, pages.ground_truth[id], pages.category[id]
            #print pred_y,label

        else:
            dist,index = sitemap.nbrs.kneighbors(x)
            print index[0]
            for id in index[0]:
                print pages.pages[id].path, pages.ground_truth[id], pages.category[id]
            print pred_y,label
    print right, total



    #file_path = "../../Crawler/Mar15_samples/stackexchange/http:__android.stackexchange.com_users_263_steffen-opel.html"

    file_path = "../../Crawler/Mar15_samples/stackexchange/http:__android.stackexchange.com_a_28782.html"

    page = Page(file_path)
    print sitemap.nbrs
    print page.path
    x = []
    for feat in sitemap.features:
        if feat in page.xpaths:
            x.append(page.xpaths[feat] * pages.idf[feat])
        else:
            x.append(0.0)

    print sitemap.nbrs.predict(x)
    '''