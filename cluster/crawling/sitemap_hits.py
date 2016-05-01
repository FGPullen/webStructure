import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from HITS import HITS
from pages import allPages
import numpy as np
from sklearn.preprocessing import normalize
import re
from collections import Counter


class sitemap:
    def __init__(self,dataset):
        self.dataset = dataset
        self.select_rules()
        self.class_xpath = {}  # {class}{xpath} = []
        self.class_xpath_link = {}
        self.gold_map_dict = self.get_gold_map_dict(self.dataset)
        self.get_clustering(self.dataset)
        self.get_trans_mat(self.dataset)
        self.write()

    def get_clustering(self, dataset):
        #path = "../clustering/new_{0}_dbscan_log-tf-idf.txt".format(dataset)
        path = "./Apr17/new_{0}.txt".format(dataset)
        lines = open(path, "r").readlines()
        self.class_dict = {}
        self.cluster_dict = {}
        for line in lines:
            [name, gold, cluster] = line.strip().split()

            gold, cluster = int(gold.replace("gold:","")), int(cluster.replace("cluster:",""))
            print name , gold, cluster
            self.class_dict[name] = gold
            self.cluster_dict[name] = cluster
            self.cluster_num = len(set(self.cluster_dict.values()))
            self.max_cluster_num = max(self.cluster_dict.values())
            self.max_class_num = max(self.class_dict.values())
            self.class_num =len(set(self.class_dict.values()))
        print self.max_cluster_num, self.max_class_num

    def get_trans_mat(self,dataset):
        self.pages = allPages(["../../Crawler/Apr17_samples/{0}/".format(dataset)],"new_{}".format(dataset), mode="raw")
        trans_mat = np.zeros((self.max_class_num+1, self.max_class_num+1))
        total_links = 0
        count_list = {}
        print len(self.pages.pages)
        for page in self.pages.pages:
            path = page.path.replace("../../Crawler","../Crawler")
            class_id = self.class_dict[path]
            cluster_id = self.cluster_dict[path]
            if class_id == -1:
                continue
                # print page.path , group
            if class_id not in count_list:
                count_list[class_id] = 1
            else:
                count_list[class_id] += 1
            link_dict = page.getAnchor()

            if cluster_id not in self.class_xpath:
                self.class_xpath[cluster_id] = {}
                self.class_xpath_link[cluster_id] = {}


            for xpath, links in link_dict.iteritems():
                # initialize add xpath to class_xpath
                if xpath not in self.class_xpath[cluster_id]:
                    self.class_xpath[cluster_id][xpath] = []
                    self.class_xpath_link[cluster_id][xpath] = []

                for link in links:
                    if self.check_intralink(link):
                        tag = self.annotate(link,self.gold_map_dict)

                        if tag != -1:
                            total_links += 1
                            trans_mat[class_id, tag] += 1
                            self.class_xpath[cluster_id][xpath].append(tag)
                            self.class_xpath_link[cluster_id][xpath].append(link)

                        # if group == tag:
                    #	print link , page.path
                    # print trans_mat
                    # trans_mat = normalize(trans_mat,norm='l1', axis=1)
        print count_list
        for i in range(self.max_class_num+1):
            for j in range(self.max_class_num+1):
                if i not in count_list:
                    trans_mat[i,j] = 0
                else:
                    trans_mat[i, j] = float(trans_mat[i, j]) / float(count_list[i])

        print "total_links has " + str(total_links)
        self.trans_mat = trans_mat

    def select_rules(self):
        stackexchange_rules = [["a","^[0-9]+.html$"],["feeds"],["help","badges"],["help","priviledges"],["posts","^[0-9]+$", "edit.html"] , ["posts","^[0-9]+$","revisions.html"],\
        ["q","^[0-9]+.html$"],["questions","^[0-9]+$"],["questions","tagged"], ["revisions","view-source.html"], ["^search?(.*)$"], ["tags"],["users","^[0-9]+(.*)$"],\
        ["users","^signup?(.*)$"]]

        douban_rules = [["awards"],["celebrity","^[0-9]+$"],["feed","subject","^[0-9]+$"],["photos","photo"],["review","^[0-9]+$"],\
                ["subject","^[0-9]+$"],["ticket","^[0-9]+$"],["trailer","^[0-9]+$"]]

        youtube_rules = [["channel","^(.*).html$"],["^playlist\?list=(.*)$"],["user","^playlists(.*).html$"],["user","^videos(.*).html$"],\
                         ["user","^discussion(.*).html$"],["user","^(.*).html$"],["^watch\?v=(.*)$"]]

        baidu_rules = [["bawu2","platform","^detailsInfo(.*)$"],["bawu2","platform","^listMemberInfo(.*)$"],["^f\?ie=utf-8(.*)$"],["f","^good\?kw(.*)$"],["f","index"],["^f\?kw(.*)$"],["f","like"],["game","^index?(.*)$"],["home","^main(.*)$"],["p","^[0-9]+(.*)$"],["shipin","bw"],\
               ["sign","^index(.*)$"],["tousu","new","^add(.*)$"]]

        tripadvisor_rules = [["^AllLocations(.*)$"],["^Attractions(.*)$"],["^Flights(.*)$"],["Hotel","^Review-(.*)$"],["^Hotels-(.*)$"],["^HotelsList-(.*)$"],["^HotelsNear-(.*)$"]\
                     ,["^LastMinute(.*)$"],["^LocalMaps(.*)$"],["^Offers(.*)$"],["^Restaurants(.*)$"],["^ShowForum(.*)$"],\
                     ["^ShowUserReviews(.*)$"],["^Tourism(.*)$"],["Travel","^Guide(.*)$"],["^TravelersChoice(.*)$"],["^VacationRentals(.*)$"],["UserReview-e"]]
        asp_rules = [["^[0-9]+.aspx$"],["f","rss"],["f","topanswerers"],["f"],["login","^RedirectToLogin?(.*)$"],["members"],["p","^[0-9]+$"]\
             ,["post","^[0-9]+.aspx.html$"],["private-message"],["^search?(.*)$"],["t","^[0-9]+.aspx(.*)$"],["t","next","^[0-9]+.html$"],["t","prev","^[0-9]+.html$"]\
             ,["t","rss","^[0-9]+.html$"]]

        if self.dataset == "stackexchange":
            self.rules = stackexchange_rules
        elif self.dataset == "douban":
            self.rules = douban_rules
        elif self.dataset == "youtube":
            self.rules = youtube_rules
        elif self.dataset == "baidu":
            self.rules = baidu_rules
        elif self.dataset == "tripadvisor":
            self.rules = tripadvisor_rules
        elif self.dataset == "asp":
            self.rules = asp_rules

    def get_gold_map_dict(self, dataset):
        path = "./site.gold/{0}/{0}.combine".format(dataset)
        map_file = open(path,"r").readlines()
        map_dict = {}
        for line in map_file:
            [value, key, help] = line.strip().split()
            map_dict[int(key)] = int(value)
        print map_dict
        return map_dict



    def annotate(self, path , gold_map_dict):
        flag = 0
        for index,rule in enumerate(self.rules):
            #print url,rule
            if self.match(path,rule):
                tag = index
                flag = 1
                break
        if flag == 0:
            tag = -1
        # remaptag
        if tag in gold_map_dict:
            tag = gold_map_dict[tag]

        return tag

    def match(self, url, rule):
        strip_url = url.strip()
        temp, terms = strip_url.replace("_","/").split("/"), []
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


    def check_intralink(self,link):
        if self.dataset == "douban":
            if "movie.douban" in link:
                return 1
            else:
                if "https" not in link:
                    return 1
                else:
                    return 0
        else:
            if "https" in link:
                return 0
            else:
                return 1

    # def compute_hits(self):
    def write(self):
        file = open("./Transition_class/{0}.mat".format(self.dataset), "w")
        for i in range(self.max_class_num+1):
            for j in range(self.max_class_num+1):
                if i == j:
                    file.write(str(i) + " " + str(j) + " " + str(0) + "\n")
                else:
                    file.write(str(i) + " " + str(j) + " " + str(self.trans_mat[i, j]) + "\n")


    def analyze_xpaths(self):
        #self.class_xpath
        write_file = open("cluster_xpaths_{0}.txt".format(self.dataset),"w")
        write_file_2 = open("cluster_xpaths_links_{0}.txt".format(self.dataset),"w")
        for cluster_id in self.class_xpath:
            if cluster_id == -1:
                continue
            xpath_dict = self.class_xpath[cluster_id]
            xpath_link_dict = self.class_xpath_link[cluster_id]
            #print xpath_dict
            for xpath in xpath_dict:
                c = Counter(xpath_dict[xpath])
                c_link = Counter(xpath_link_dict[xpath])
                print cluster_id, xpath, c
                print cluster_id, xpath, c_link
                write_file.write(str(cluster_id) + "\t" + xpath + "\t" + str(c)+"\n")
                write_file_2.write(str(cluster_id) + "\t" + xpath + "\t" + str(c_link)+"\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",choices=["stackexchange","douban","youtube","baidu","tripadvisor","asp"],help="dataset for computing transition matrix")
    args = parser.parse_args()
    s = sitemap(args.dataset)
    #s.analyze_xpaths()
