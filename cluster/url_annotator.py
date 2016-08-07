import re
import os
import collections

class annotator():

    def __init__(self,dataset):
        self.dataset = dataset
        self.rules = self.get_rules(dataset)
        self.combine_map = self.get_combine_map(dataset)

    def get_rules(self,dataset):
        if dataset == "stackexchange":
            rules = [["a","^[0-9]+(.*)$"],["feeds"],["help","badges"],["help","priviledges"],["posts","^[0-9]+$", "edit"] , ["posts","^[0-9]+$","revisions"],\
            ["q","^[0-9]+(.*)$"],["questions","^[0-9]+(.*)$"],["questions","tagged"], ["revisions","view-source.html"], ["^search?(.*)$"], ["tags"],["users","^[0-9]+(.*)$"],\
            ["users","^signup?(.*)$"],["users","^login?(.*)$"]]
        elif dataset == "asp":
            rules = [["^[0-9]+.aspx$"],["f","rss"],["f","topanswerers"],["f"],["login","^RedirectToLogin?(.*)$"],["members"],["p","^[0-9]+$"]\
             ,["post","^[0-9]+.aspx.html$"],["private-message"],["^search?(.*)$"],["t","^[0-9]+.aspx(.*)$"],["t","next","^[0-9]+(.*)$"],["t","prev","^[0-9]+(.*)$"]\
             ,["t","rss","^[0-9]+(.*)$"]]
        elif dataset == "youtube":
            rules = [["channel","^(.*)$"],["^playlist\?list=(.*)$"],["user","^playlists(.*)$"],["user","^videos(.*)$"],["user","^discussion(.*)$"],["user","^(.*)$"],["^watch\?v=(.*)$"]]
        elif dataset == "douban":
            rules = [["awards"],["celebrity","^[0-9]+$"],["feed","subject","^[0-9]+$"],["photos","photo"],["review","^[0-9]+$"],\
                ["subject","^[0-9]+$"],["ticket","^[0-9]+$"],["trailer","^[0-9]+$"]]
        elif dataset == "tripadvisor":
            rules = [["^AllLocations(.*)$"],["^Attractions(.*)$"],["^Flights(.*)$"],["Hotel","^Review-(.*)$"],["^Hotels-(.*)$"],["^HotelsList-(.*)$"],["^HotelsNear-(.*)$"]\
                     ,["^LastMinute(.*)$"],["^LocalMaps(.*)$"],["^Offers(.*)$"],["^Restaurants(.*)$"],["^ShowForum(.*)$"],\
                     ["^ShowUserReviews(.*)$"],["^Tourism(.*)$"],["Travel","^Guide(.*)$"],["^TravelersChoice(.*)$"],["^VacationRentals(.*)$"],["UserReview-e"]]
        elif dataset == "rottentomatoes":
            rules = [["browse"],["celebrity","pictures"],["celebrity"],["critic"],["critics"],["guides"],["m"+"^[0-9]+$"+"pictures"],["m","trailers"],["m","reviews"],\
                    ["m"],["tv"+"^[0-9]+$"+"pictures"],["tv","trailers"],["tv","reviews"],\
                    ["tv"],["showtimes"],["^source-[0-9]+$"],["top"],["user","^[0-9]+$"]]
        elif dataset == "baidu":
            rules = [["bawu2","platform","^detailsInfo(.*)$"],["bawu2","platform","^listMemberInfo(.*)$"],["^f\?ie=utf-8(.*)$"],["f","^good\?kw(.*)$"],["f","index"],["^f\?kw(.*)$"],["f","like"],["game","^index?(.*)$"],["home","^main(.*)$"],["p","^[0-9]+(.*)$"],["shipin","bw"],\
                    ["sign","^index(.*)$"],["tousu","new","^add(.*)$"]]
        elif dataset == "biketo":
            rules = [["^forum-[0-9]+-[0-9]+(.*)$"],["^space-uid-[0-9]+(.*)$"],["^thread-[0-9]+-(.*)$"],["columns","^[0-9]+$"],["daily","^[0-9]+(.*)$"],\
               ["c?aid=[0-9]+(.*)"],["edge","beginner","index"],["edge","health","^[0-9]+$(.*)"],["edge","knowledge","^[0-9]+$(.*)"],\
                ["edge","Photographic","^[0-9]+(.*)$"],["edge","repair","^[0-9]+(.*)$"],["edge","safe","^[0-9]+(.*)$"],["Gallery","photograph","^[0-9]+(.*)$"],\
                ["industry","business","^[0-9]+(.*)$"],["industry","cover","^[0-9]+(.*)$"],["industry","exhibition","^[0-9]+(.*)$"],["info"],["news","activity","^[0-9]+(.*)$"],["news","bikenews","^[0-9]+(.*)$"],\
                ["news","girl","^[0-9]+(.*)$"],["international","^[0-9]+(.*)$"],["news","picture","^[0-9]+(.*)$"],["product","bikes","index"],["product","bikes","^[0-9]+(.*)$"],["product","equipment","^[0-9]+(.*)$"],\
                ["gearest","^[0-9]+(.*)$"],["racing","column","^[0-9]+(.*)$"],["racing","cover"],["racing","Events","^[0-9]+(.*)$"],["racing","herald","^[0-9]+(.*)$"],\
                ["racing","index"],["racing","internal","^[0-9]+(.*)$"],["racing","news","^[0-9]+(.*)$"],["s"],["z","all","^[0-9]+(.*)$"],["z","^[0-9]+(.*)$"]]
        elif dataset == "hupu":
            rules = [["cba","^[0-9]+(.*)$"],["china","^[0-9]+(.*)$"],["f1","^[0-9]+(.*)$"],["other","^[0-9]+(.*)$"],["soccer","^[0-9]+(.*)$"],["sports","^[0-9]+(.*)$"],\
              ["tennis","^[0-9]+(.*)$"],["zb","^[0-9]+(.*)$"],["nba","^[0-9]+(.*)$"],["o"],["people","^[0-9]+(.*)$"]]

        return rules

    def get_combine_map(self,dataset):
        if dataset == "stackexchange":
            combine_map = [(0,6),(0,7)]
        elif dataset == "asp":
            combine_map = [(0,3),(4,8),(6,7),(6,10),(6,11),(6,12)]
        elif dataset == "youtube":
            combine_map = [(0,2),(0,3),(0,4),(0,5)]
        elif dataset == "douban":
            combine_map = []
        elif dataset == "tripadvisor":
            combine_map = []
        elif dataset == "rottentomatoes":
            combine_map = [(9,13),(4,8)]
        elif dataset == "baidu":
            combine_map = [(3,2),(3,5)]
        elif dataset == "hupu":
            combine_map = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(9,10)]
        mapping = {}
        for map in combine_map:
            mapping[map[1]] = map[0]

        return mapping


    def get_ground_truth(self,url_list):
        rules = self.rules
        class_list = []
        for url in url_list:
            flag = 0
            for index,rule in enumerate(rules):
                #print url,rule
                if self.match(url,rule):
                    if index in self.combine_map:
                        class_list.append(self.combine_map[index])
                    else:
                        class_list.append(index)
                    flag = 1
                    break
            if flag == 0:
                class_list.append(-1)
        assert len(class_list) == len(url_list)
        #for i in range(len(url_list)):
        #    print class_list[i], url_list[i]
        return  class_list

    def match(self,url,rule):
        strip_url = url.strip().replace("/","_")
        temp, terms = strip_url.split("_"), []
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

    def annotate_file(self,file_path):
        url_list = []
        lines = open(file_path,'r').readlines()
        for line in lines:
            tmp = line.strip().split()
            if len(tmp) == 1:
                tmp = line.strip().split("\t")
            url_list.append(tmp[0])
        #print url_list
        print len(url_list), " length of url list"

        class_list = self.get_ground_truth(url_list)
        #for index, url in enumerate(url_list):
        #    print url, class_list[index]
        self.url_list = url_list
        self.class_list = class_list

        t = collections.Counter()
        t.update(class_list)

        return t


    def output_results(self):
        print "length of url_list is {}".format(len(self.url_list))
        print "length of class_list is {}".format(len(self.class_list))
        length = len(self.url_list)
        for i in range(length):
            if self.class_list[i] == -1:
                print self.class_list[i], self.url_list[i]
        c = collections.Counter(self.class_list)
        print c[-1], " the number of -1"



'''
rotten_rules = [["browse"],["celebrity","pictures"],["celebrity"],["critic"],["critics"],["guides"],["m"+"^[0-9]+$"+"pictures"],["m","trailers"],["m","reviews"],\
["m"],["tv"+"^[0-9]+$"+"pictures"],["tv","trailers"],["tv","reviews"],\
["tv"],["showtimes"],["^source-[0-9]+$"],["top"],["user","^[0-9]+$"]]




tripadvisor_rules = [["^AllLocations(.*)$"],["^Attractions(.*)$"],["^Flights(.*)$"],["Hotel","^Review-(.*)$"],["^Hotels-(.*)$"],["^HotelsList-(.*)$"],["^HotelsNear-(.*)$"]\
                     ,["^LastMinute(.*)$"],["^LocalMaps(.*)$"],["^Offers(.*)$"],["^Restaurants(.*)$"],["^ShowForum(.*)$"],\
                     ["^ShowUserReviews(.*)$"],["^Tourism(.*)$"],["Travel","^Guide(.*)$"],["^TravelersChoice(.*)$"],["^VacationRentals(.*)$"],["UserReview-e"]]

biketo_rules = [["^forum-[0-9]+-[0-9]+(.*)$"],["^space-uid-[0-9]+(.*)$"],["^thread-[0-9]+-(.*)$"],["columns","^[0-9]+$"],["daily","^[0-9]+(.*)$"],\
               ["c?aid=[0-9]+(.*)"],["edge","beginner","index"],["edge","health","^[0-9]+$(.*)"],["edge","knowledge","^[0-9]+$(.*)"],\
                ["edge","Photographic","^[0-9]+(.*)$"],["edge","repair","^[0-9]+(.*)$"],["edge","safe","^[0-9]+(.*)$"],["Gallery","photograph","^[0-9]+(.*)$"],\
                ["industry","business","^[0-9]+(.*)$"],["industry","cover","^[0-9]+(.*)$"],["industry","exhibition","^[0-9]+(.*)$"],["info"],["news","activity","^[0-9]+(.*)$"],["news","bikenews","^[0-9]+(.*)$"],\
                ["news","girl","^[0-9]+(.*)$"],["international","^[0-9]+(.*)$"],["news","picture","^[0-9]+(.*)$"],["product","bikes","index"],["product","bikes","^[0-9]+(.*)$"],["product","equipment","^[0-9]+(.*)$"],\
                ["gearest","^[0-9]+(.*)$"],["racing","column","^[0-9]+(.*)$"],["racing","cover"],["racing","Events","^[0-9]+(.*)$"],["racing","herald","^[0-9]+(.*)$"],\
                ["racing","index"],["racing","internal","^[0-9]+(.*)$"],["racing","news","^[0-9]+(.*)$"],["s"],["z","all","^[0-9]+(.*)$"],["z","^[0-9]+(.*)$"]]


hupu_rules = [["cba","^[0-9]+(.*)$"],["china","^[0-9]+(.*)$"],["f1","^[0-9]+(.*)$"],["other","^[0-9]+(.*)$"],["soccer","^[0-9]+(.*)$"],["sports","^[0-9]+(.*)$"],\
              ["tennis","^[0-9]+(.*)$"],["zb","^[0-9]+(.*)$"],["nba","^[0-9]+(.*)$"],["o"],["people","^[0-9]+(.*)$"]]


baidu_rules = [["bawu2","platform","^detailsInfo(.*)$"],["bawu2","platform","^listMemberInfo(.*)$"],["^f\?ie=utf-8(.*)$"],["f","^good\?kw(.*)$"],["f","index"],["^f\?kw(.*)$"],["f","like"],["game","^index?(.*)$"],["home","^main(.*)$"],["p","^[0-9]+(.*)$"],["shipin","bw"],\
               ["sign","^index(.*)$"],["tousu","new","^add(.*)$"]]
'''

if __name__ == "__main__":
    site = "stackexchange"
    a = annotator(site)
    #file_path = "./results/sampling/random_uniform_{0}_size1001.txt".format(site)
    #file_path = "../May1/site.dbscan/{}.txt".format(site)
    file_path =  "./July30/site.sample/{}.sample".format(site)
    a.annotate_file(file_path)
    a.output_results()