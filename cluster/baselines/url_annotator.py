import re
import os,sys
import collections
import traceback

class annotator():

    def __init__(self,dataset):
        self.dataset = dataset
        self.rules = self.get_rules(dataset)
        self.combine_map = self.get_combine_map(dataset)

    def get_rules(self,dataset):
        if dataset == "stackexchange":
            rules = [["a","^[0-9]+(.*)$"],["feeds"],["help","badges"],["help","priviledges"],["posts","^[0-9]+$", "edit"] , ["posts","^[0-9]+$","revisions"],\
            ["q","^[0-9]+(.*)$"],["questions","^[0-9]+(.*)$"],["questions","tagged"], ["revisions","view-source.html"], ["^search?(.*)$"], ["tags"],["users","^(.*)?tab=profile$"],["users","^(.*)?tab=(.*)$"],["users","^[0-9]+(.*)$"],\
            ["users","^signup?(.*)$"],["users","^login?(.*)$"],["^questions?(.*)=(.*)$"],["unanswered","tagged"],["help","^(.*)$"],["^users?(.*)=(.*)$"]]
        elif dataset == "asp":
            rules = [["f","rss"],["f","topanswerers"],["f"],["login","^RedirectToLogin?(.*)$"],["members"],["p","^[0-9]+$"]\
             ,["post","^[0-9]+.aspx$"],["private-message"],["^search?(.*)$"],["t","^[0-9]+.aspx(.*)$"],["t","next","^[0-9]+(.*)$"],["t","prev","^[0-9]+(.*)$"]\
             ,["t","rss","^[0-9]+(.*)$"],["^[0-9]+.aspx$"]]
        elif dataset == "youtube":
            rules = [["channel","^(.*)$"],["^playlist\?list=(.*)$"],["user","^playlists(.*)$"],["user","^videos(.*)$"],["user","^discussion(.*)$"],["user","^(.*)$"],["^watch\?v=(.*)$"]]
        elif dataset == "douban":
            rules = [["accounts","sina"],["accounts","wechat"],["accounts","^login?(.*)$"],["awards"],["celebrity","detail","edit"],["celebrity","^[0-9]+$"],["feed","subject","^[0-9]+$"],["photos","photo"],["review","^[0-9]+$"],\
                ["subject","questions","ask"],["ticket","^[0-9]+$"],["subject","^create?(.*)$"],["subject","^[0-9]+$","cinema"],["subject","^[0-9]+$","photos"],["subject","^[0-9]+$","questions"],["subject","^[0-9]+$","^doing|wishes|collections$"],\
                     ["subject","^[0-9]+$","^photos?(.*)$"],["subject","^[0-9]+$","^follows?(.*)$"],["subject","^[0-9]+$","^reviews?(.*)$"],\
                     ["subject","^[0-9]+$","^doulist?(.*)$"],["subject","^[0-9]+$","^trailer?(.*)$"],["subject","^[0-9]+$","^comments?(.*)$"],\
                     ["subject","^[0-9]+$","discussion"],["subject","^[0-9]+$"],["trailer","^[0-9]+$"]]
        elif dataset == "tripadvisor":
            rules = [["^AllLocations(.*)$"],["^Attractions(.*)$"],["Attraction","^Review-(.*)$"],["Restaurant","^Review-(.*)$"],["^Flights(.*)$"],["Hotel","^Review-(.*)$"],["^Hotels-(.*)$"],["^HotelsList-(.*)$"],["^HotelsNear-(.*)$"]\
                     ,["^LastMinute(.*)$"],["^LocalMaps(.*)$"],["^Offers(.*)$"],["^Restaurants(.*)$"],["^ShowForum(.*)$"],\
                     ["^Tourism(.*)$"],["Travel","^Guide(.*)$"],["^TravelersChoice(.*)$"],["^VacationRentals(.*)$"],["^VacationRentalReview(.*)$"],\
                     ["FAQ","^Answers-(.*)$"],["^LocationPhotoDirectLink(.*)$"],["^PressCenter-(.*)$"],["^ManagementCenter-(.*)$"],["^UserReviewEdit(.*)$"],["^UserReview(.*)$"],["^ShowUserReviews(.*)$"],["^SmartDeals-(.*)$"],["^MediaKit\?(.*)$"],["^ShowTopic-(.*)$"],["Vacation","^Packages-(.*)$"],["^PostPhotos-(.*)$"]]
        elif dataset == "rottentomatoes":
            rules = [["browse"],["celebrity","pictures"],["celebrity"],["critic"],["critics"],["guides"],["m","pictures"],["m","trailers"],["m","reviews"]\
                    ,["m","quotes"],["m"],["tv","pictures"],["tv","trailers"],["tv","reviews"],["tv","videos"],\
                    ["tv"],["showtimes"],["^source-[0-9]+$"],["top"],["user","^[0-9]+$"],["help","desk"]]
        elif dataset == "baidu":
            rules = [["bawu2","platform","^detailsInfo(.*)$"],["bawu2","platform","^listMemberInfo(.*)$"],["^f\?ie=utf-8(.*)$"],["f","^good\?kw(.*)$"],["f","index"],["^f\?kw(.*)$"],["f","like"],["game","^index?(.*)$"],["home","^main(.*)$"],["p","^[0-9]+(.*)$"],["shipin","bw"],\
                    ["sign","^index(.*)$"],["tousu","new","^add(.*)$"],["c","s","download","^pc/?src=(.*)$"],["photo","^g\?kw=(.*)$"],["im","^pcmsg\?from=(.*)$"],["home","^achievement\?un=(.*)$"],["tbmall","^tshow\?tab=(.*)$"],["newvote","^createvote\?kw=(.*)$"],["tbmall","gift","^detail\?gift(.*)$"],\
                     ["show","zhanqi","^roomList\?tag$"],["platform","agency"],["^tieba.baidu.com#(.*)$"],["bawu2","platform","^listCandidateInfo\?word=(.*)$"],["f","^fdir\?fd=(.*)$"],["home","^fans\?id(.*)|concern\?id(.*)$"],["tbmall","^propslist\?category=(.*)$"]]
        elif dataset == "huffingtonpost":
            rules = [["^(.*)$","^ref=(.*)$"],["news","^(.*)$"]]
        elif dataset == "photozo":
            rules = [["members","list"],["members"],["^member.php\?(.*)$"],["forum","blogs","^(.*)$","^((?!feed.rss).)*$"],["^subscription.php\?do=addsubscription&t=(.*)$"],["^newreply.php\?do=(.*)$"],\
                     ["^forumdisplay.php\?do=markread\&markreadhash=(.*)$"],["^search.php\?(.*)$"],["^sendmessage.php\?do(.*)$"],["^awards.php#(.*)$"],["^blog.php\?u=(.*)\&do=markread\&readhash=(.*)$"],["^event-photography|groups|video-photography|lens|knowledge-base|announcement|sponsors-special-deals|world-travel|lighting-technique|sports|photography-basics|nature-wildlife-animals|digital-manipulation|critique|cameras|people|photo-assignments|introductions-welcomes|other-misc|fashion-glamour-artistic-nude$","^#(.*)|(.*)\?sort=(.*)|index(.*)$"],\
                     ["^event-photography|groups|video-photography|lens|knowledge-base|announcement|world-travel|lighting-technique|sports|photography-basics|nature-wildlife-animals|digital-manipulation|critique|cameras|people|photo-assignments|introductions-welcomes|other-misc|sponsors-special-deals|fashion-glamour-artistic-nude$","^(.*)$"],["^misc.php\?do(.*)$"],\
                     ["^threadtag.php\?do(.*)$"],["^entry.php\?b(.*)\&do=sendtofriend$"],["^external.php\?type=RSS2\&forumids=(.*)$"],["^event-photography|groups|video-photography|lens|knowledge-base|announcement|sponsors-special-deals|world-travel|lighting-technique|sports|photography-basics|nature-wildlife-animals|digital-manipulation|critique|cameras|people|photo-assignments|introductions-welcomes|other-misc|fashion-glamour-artistic-nude$","^#(.*)$"]]
        elif dataset == "hupu":
            rules = [["newslist"],["cba","^[0-9]+(.*)$"],["china","^[0-9]+(.*)$"],["f1","^[0-9]+(.*)$"],["other","^[0-9]+(.*)$"],["soccer","^[0-9]+(.*)$"],["sports","^[0-9]+(.*)$"],\
              ["tennis","^[0-9]+(.*)$"],["zb","^[0-9]+(.*)$"],["nba","^[0-9]+(.*)$"],["wcba","^[0-9]+(.*)$"],["o"],["people","^[0-9]+(.*)$"],["player"],["topic"],["hot"]]

        return rules

    def get_combine_map(self,dataset):
        '''
        if dataset == "stackexchange":
            combine_map = [(0,6),(0,7),(8,17),(8,18),(12,14)]
        elif dataset == "asp":
            combine_map = [(2,13),(3,7),(5,6),(5,9),(5,10),(5,11)]
        elif dataset == "youtube":
            combine_map = [(0,2),(0,3),(0,4),(0,5)]
        elif dataset == "douban":
            combine_map = [(4,2),(4,9),(4,10),(4,11)]
        elif dataset == "tripadvisor":
            combine_map = []
        elif dataset == "rottentomatoes":
            combine_map = [(10,15),(4,8),(4,13),(1,6),(1,11),(7,14)]
        elif dataset == "baidu":
            combine_map = [(3,2),(3,5),(8,16)]
        elif dataset == "hupu":
            combine_map = [(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(1,10),(11,12)]
        elif dataset == "huffingtonpost":
            combine_map =[]
        elif dataset == "photozo":
            combine_map =[(1,2),(4,5),(4,6),(4,7),(11,18)]
        mapping = {}
        for map in combine_map:
            mapping[map[1]] = map[0]
        '''
        return []


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
                    #traceback.print_exc()
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
        c_list, u_list = zip(*sorted(zip(self.class_list, self.url_list)))
        for i in range(length):
            #if c_list[i] == 1 or c_list[i] == 4 or c_list[i] == 7 or c_list[i] == 9 or c_list[i] == 12:
            print c_list[i],u_list[i]
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

    site = "hupu"
    a = annotator(site)
    #file_path = "./results/sampling/random_uniform_{0}_size1001.txt".format(site)
    #file_path = "../May1/site.dbscan/{}.txt".format(site)
    file_path =  "./July30/site.sample/{}.sample".format(site)
    a.annotate_file(file_path)
    a.output_results()