from page import Page
from cluster import cluster
import numpy as np
import os
from sklearn.preprocessing import scale,normalize
import math
import collections
from sets import Set


class allPages:
    def __init__(self, folder_path ,dataset,mode="read"): # mode: {raw, read, write}
        self.folder_path = folder_path
        self.dataset = dataset
        print folder_path
        self.threshold = 0.004
        self.pages = []
        self.path_list = []
        self.category = [] # prediction
        self.xpaths_set = Set()
        self.ground_truth = []  # ground truth list for all pages
        self.idf = {}
        self.selected_df = {}
        self.df = {}
        if not os.path.exists("./feature/"+dataset):
            os.makedirs("./feature/"+dataset)
        if mode == "read":
            page_list = open("./feature/"+dataset+"/pages.txt","r").readlines()
            tf_idf_lines = open("./feature/"+dataset+"/tf_idf.txt","r").readlines()
            log_tf_idf_lines = open("./feature/" + dataset + "/log_tf_idf.txt","r").readlines()
            for i in range(len(page_list)):

                    pid = page_list[i].strip().split(":")[0]
                    file_path = ":".join(page_list[i].strip().split(":")[1:])
                    file_page = Page(file_path,mode="read")
                    self.path_list.append(file_path)

                    tf_idf_features = tf_idf_lines[i].strip().split(":")[-1]
                    file_page.read_tf_idf(tf_idf_features)
                    
                    log_tf_idf_features = log_tf_idf_lines[i].strip().split(":")[-1]
                    file_page.read_log_tf_idf(log_tf_idf_features)
                    
                    self.pages.append(file_page)
            
            self.category = [0 for i in range(len(page_list))]
            self.get_ground_truth(dataset)
        else:
        # initialize data structure
            #  update attributes
            self.addPages(folder_path)
            self.expandXpaths()
            self.updateidf()
            #self.get_ground_truth(dataset)
            self.num = len(self.pages)
            #self.top_local_stop_structure_gt(0.9)
            self.updatetfidf()
            self.filter_df(0.01,1.0)
            #self.filter_dfs_xpaths_list()
            #self.Leung_baseline()  # binary feature
            self.selected_tfidf()
            
            if mode=="write":
                # filtered xpath :  id xpath
                xpath_file =  open("./feature/"+dataset+"/xpaths.txt","w")
                for page in self.pages:
                    xpath_id = 0
                    for xpath in page.selected_tfidf:
                        xpath_file.write(str(xpath_id)+":"+xpath+"\n")
                        xpath_id+=1
                    break

                page_file =  open("./feature/"+dataset + "/pages.txt","w")# id file_path 
                tf_idf_file =  open("./feature/" + dataset + "/tf_idf.txt","w")  # pid features..
                log_tf_idf_file = open("./feature/" + dataset +"/log_tf_idf.txt","w")
                page_id = 0
                for page in self.pages:
                    page_file.write(str(page_id)+":"+page.path+"\n")
                    vector = []
                    for key in page.selected_tfidf:
                        vector.append(page.selected_tfidf[key])
                    tf_idf_file.write(str(page_id)+":" +" ".join(str(feat) for feat in vector) + "\n")
                    vector = []
                    for key in page.selected_logtfidf:
                        vector.append(page.selected_logtfidf[key])
                    log_tf_idf_file.write(str(page_id)+":" + " ".join(str(feat) for feat in vector)+"\n")

                    page_id += 1

        #self.update_bigram()

    def filter_dfs_xpaths_list(self):
        print "start filtering for xpaths list"
        for page in self.pages:
            page.filtered_dfs_xpaths_list = [item for item in page.dfs_xpaths_list if item in self.selected_df]
            #print "we filterd " + str(len(page.dfs_xpaths_list) - len(page.filtered_dfs_xpaths_list))
        print "end filtering for xpaths list"

    def update_xpaths_set(self,_page_):
        for xpath in _page_.xpaths.keys(): # xpaths set
            self.xpaths_set.add(xpath)

    def addPages(self,folder_path_list):
        category_num = 0
        for folder_path in folder_path_list:
            folder_pages = []
            for html_file in os.listdir(folder_path):
                if ".DS_Store" not in html_file:
                    file_path = folder_path + html_file
                    file_page = Page(file_path)
                    # the same number for pags & category
                    self.pages.append(file_page)
                    self.path_list.append(file_page.path)
                    self.category.append(category_num)
                    self.update_xpaths_set(file_page)
            category_num+=1

    def expandXpaths(self):
        for page in self.pages:
            page.expandXpaths(self.xpaths_set)

    def updateidf(self):
        N = len(self.pages)
        # initiate
        for xpath in self.xpaths_set:
            self.df[xpath] = 0
        # count document frequency
        for page in self.pages:
            for xpath in self.xpaths_set:
                if page.xpaths[xpath] !=0:
                    self.df[xpath] +=1
        # log(n/N)
        for xpath in self.xpaths_set:
            self.idf[xpath] = math.log((float(N))/(float(self.df[xpath])),2)
            # add sqrt into idf so that calculating distance there will be only one power of idf
            #self.idf[xpath] = math.sqrt(self.idf[xpath])

    def top_local_stop_structure_gt(self,threshold):
        global_threshold = len(self.pages) * threshold
        gt_clusters = []
        for item in set(self.ground_truth):
            gt_clusters.append(cluster())
            for i in range(len(self.ground_truth)):
                if self.ground_truth[i] == item:
                    gt_clusters[item].addPage(self.pages[i])
            print str(item) + "\t" + str(len(gt_clusters[item].pages))

        print "number of gt cluster is " + str(len(gt_clusters))
        print "number of cluster 5 is " + str(len(gt_clusters[4].pages))
        gt_clusters[4].find_local_stop_structure(self.df,global_threshold)

    def affinity(self,page1,page2):
        distance = 0.0
        s,t,m =0.0,0.0,0.0
        for key in self.selected_df:
            if page1.xpaths[key] >= 1:
                s += 1
            if page2.xpaths[key] >= 1:
                t += 1
            if page1.xpaths[key] >= 1 and page2.xpaths[key] >= 1:
                m += 1
        if s+t-m == 0:
            distance = 0.0
        else:
            distance = float(m)/float(s+t-m)
        return distance

    def get_affinity_matrix(self):
        # calculate affinity matrix based on Xpath
        # if we assign a matrix before , would it be much quicker?
        print len(self.selected_df)
        matrix = np.zeros(shape=(self.num,self.num))
        for i in range(self.num):
            for j in range(i+1,self.num):
                matrix[i,j] = self.affinity(self.pages[i],self.pages[j])
                matrix[j,i] = matrix[i,j]
        return matrix

    def get_one_hot_distance_matrix(self):
        matrix = np.zeros(shape=(self.num,self.num))
        for i in range(self.num):
            temp = []
            for key, value in self.pages[i].Leung.iteritems():
                temp.append(value)
            s_i  = np.array(normalize(temp,norm='l1')[0])
            for j in range(i+1,self.num):
                temp = []
                for key, value in self.pages[j].Leung.iteritems():
                    temp.append(value)
                s_j  = np.array(normalize(temp,norm='l1')[0])
                matrix[i,j] =  np.linalg.norm(s_i - s_j)
                matrix[j,i] = matrix[i,j]
        return matrix

    def examine_one_xpath(self,xpath):
        # output the page that contain exact xpath
        for page in self.pages:
            if xpath in page.xpaths and page.xpaths[xpath]>0:
                print page.path


    def updatetfidf(self):
        for page in self.pages:
            page.updatetfidf(self.idf)

    # update category based on predicted y
    def updateCategory(self,pred_y):
        assert len(self.category) == len(pred_y)
        for i in range(len(pred_y)):
            self.category[i] = pred_y[i]

    def get_edit_distance_matrix(self):
        matrix = np.zeros(shape=(self.num,self.num))
        for i in range(self.num):
            if i % 8 == 0:
                print "finish " + str(float(i)/float(self.num)) + "% of " + str(self.num) + " pages"
            s_i = self.pages[i].filtered_dfs_xpaths_list
            for j in range(i+1,self.num):
                    s_j = self.pages[j].filtered_dfs_xpaths_list
                    matrix[i][j] = int(distance.levenshtein(s_i,s_j))
                    matrix[j][i] = matrix[i][j]
                    print str(i) + "\t" + str(j)
                    #print str(len(s_i)) + "\t" + str(len(s_j))
        return matrix

    def read_edit_distance_matrix(self):
        matrix = np.zeros(shape=(self.num,self.num))
        lines = open("./Data/edit_distance.matrix","r").readlines()
        for i in range(self.num):
            line = lines[i].strip().split("\t")
            assert len(line) == self.num
            for j,item in enumerate(line):
                matrix[i][j] = float(item)
        return matrix

    def build_gold(self,lines):
        gold_dict = {}
        for line in lines:
            line = line.strip()
            if line!="":
                [path,id] = line.split()
                path = path.replace("_","/").replace(".html","")
                gold_dict[path] = id
        return gold_dict

    def get_ground_truth(self,dataset):
        if dataset == "new_stackexchange":
            gold_file = open("./Annotation/site.gold/stackexchange/stackexchange.gold").readlines()
            gold_dict = self.build_gold(gold_file)
            #print gold_dict.keys()[0]
            for i in range(len(self.pages)):
                path = self.pages[i].path.replace("../Crawler/Mar15/samples/stackexchange/","")
                id = int(gold_dict[path])
                self.ground_truth.append(id)
        elif dataset == "new_rottentomatoes":
            gold_file = open("./Annotation/site.gold/rottentomatoes/rottentomatoes.gold").readlines()
            gold_dict = self.build_gold(gold_file)
            print self.folder_path
            for i in range(len(self.pages)):
                path = self.pages[i].path.replace("../Crawler/Mar15/samples/rottentomatoes/","")
                id = int(gold_dict[path])
                self.ground_truth.append(id)
        elif dataset == "new_biketo":
            gold_file = open("./Annotation/site.gold/biketo/biketo.gold").readlines()
            gold_dict = self.build_gold(gold_file)
            print self.folder_path
            for i in range(len(self.pages)):
                path = self.pages[i].path.replace("../Crawler/Mar15/samples/biketo/","")
                if ".jpg.html" in path:
                    print path
                    id = -1
                else:
                    id = int(gold_dict[path])
                self.ground_truth.append(id)
        elif "new_" in dataset:
            data = dataset.replace("new_","")
            gold_file = open("./Annotation/site.gold/{0}/{0}.gold".format(data)).readlines()
            gold_dict = self.build_gold(gold_file)
            #print self.folder_path
            for i in range(len(self.pages)):
                path = self.pages[i].path.replace("../Crawler/Mar15/samples/{}/".format(data),"")
                id = int(gold_dict[path.strip()])
                self.ground_truth.append(id)
        else:
            # /users/ /questions/ /q/ /questions/tagged/   /tags/ /posts/ /feeds/ /others
            if "../Crawler/crawl_data/Questions/"  in self.folder_path or "../Crawler/test_data/train/" in self.folder_path or "../Crawler/test_data/stackexchange/" in self.folder_path:
                print "????"
                for i in range(len(self.pages)):
                    path = self.pages[i].path.replace("../Crawler/crawl_data/Questions/", "")
                    if "/users/" in path:
                        tag = 1
                    elif "/questions/tagged/" in path:
                        tag = 3
                    elif "/questions/" in path or "/q/" in path or "/a/" in path:
                        tag = 2
                    #elif "/tags/" in path:
                    #    tag = 6
                    elif "/posts/" in path:
                        tag = 5
                    elif "/feeds/" in path:
                        tag = 4
                    #else:
                    #    tag = 0
                    #print "tag is " + str(tag)
                    self.ground_truth.append(tag)
            # zhihu
            # /people/  /question/ /question/answer/ /topic/  (people/followed/ people/follower/ -> index ) /ask /collection
            elif "../Crawler/crawl_data/Zhihu/" in self.folder_path or "../Crawler/test_data/zhihu/" in self.folder_path:
                print "!!!!"
                for i in range(len(self.pages)):
                    path = self.pages[i].path.replace("../Crawler/test_data/zhihu/","")
                    if "follow" in path:
                        tag = 2
                    elif "/people/" in path:
                        tag = 0
                    elif "/question/" in path:
                        tag = 1
                    elif "/topic/" in path:
                        tag = 3
                    elif "/collection/" in path:
                        tag = 4
                    else:
                        tag =5
                    self.ground_truth.append(tag)
            elif "../Crawler/test_data/rottentomatoes/" in self.folder_path:
                print "rottentomatoes datasets"
                for i in range(len(self.pages)):
                    path = self.pages[i].path.replace("../Crawler/test_data/rottentomatoes/","")
                    if "/top/" in path:
                        tag = 2
                    elif "/guides/" in path:
                        tag = 5
                    elif "/celebrity/" in path:
                        if "/pictures/" in path:
                            tag = 6
                        else:
                            tag = 0
                    elif "/critic/" in path:
                            tag = 1
                    elif "/m/" in path or "/tv/" in path:
                        if "/trailers/" in path:
                            tag = 4
                        elif "/pictures/" in path:
                            tag = 6
                        else:
                            tag = 3
                    else: # guide
                        print path
                        tag =0
                    self.ground_truth.append(tag)

            elif "../Crawler/test_data/medhelp/" in self.folder_path or "../Crawler/test_data/test" in self.folder_path:
                print "medhelp datasets"
                for i in range(len(self.pages)):
                    path = self.pages[i].path.replace("../Crawler/test_data/medhelp/","")
                    if "/forums/" in path:
                        tag = 2
                    elif "/groups/" in path:
                            tag = 2
                    elif "/personal/" in path:
                            tag = 1
                    elif "/posts/" in path:
                        tag = 3

                    elif "/tags/" in path:
                        tag =4
                    else:
                        tag = 5
                    self.ground_truth.append(tag)

            elif "../Crawler/test_data/ASP/" in self.folder_path:
                print "asp.net datasets"
                for i in range(len(self.pages)):
                    path = self.pages[i].path.replace("../Crawler/test_data/ASP/","")
                    if "/f/" in path:
                        if "/topanswerers/" in path:
                            print path
                            tag = 5
                        else:
                            tag = 2
                    elif "/members/" in path:
                        tag = 0
                    elif "RedirectToLogin" in path or "/private-message/" in path:
                        tag = 1
                    elif "/post/" in path:
                        tag = 3
                    elif "/t/" in path or "/p/" in path:
                        tag =3
                    elif "search?" in path:
                        tag =4
                    else:
                        tag = 2
                    self.ground_truth.append(tag)

    def Leung_baseline(self):
        # one-hot representation
        # threshold set to be 0.25 which means that xpath appear over 25% pages will be kept.
        N = self.num
        for key in self.idf:
            #print key
            if float(self.df[key])/float(N) >= self.threshold:
                for page in self.pages:
                    page.update_Leung(key)

    def update_bigram(self):
        bigram_list = self.find_bigram()
        for page in self.pages:
            page.get_bigram_features(bigram_list)
            #print len(page.bigram_dict)


    def filter_df(self,min_t, max_t):
        # aim to get a subset of xpaths which filters out path that appears in very few documents
        # as well as those that occur in almost all documents
        for key in self.idf:
            if float(self.df[key])/float(self.num) >= min_t:
                if float(self.df[key])/float(self.num) <= max_t:
                    self.selected_df[key] = self.df[key]

    def selected_tfidf(self):
        N = self.num
        for key in self.idf:
            if float(self.df[key])/float(N) >= self.threshold:
                for page in self.pages:
                    page.update_selected_tfidf(key)

    def find_important_xpath(self):
        length = len(self.pages)
        print "numer of pages " + str(length)
        print "number of xpath " + str(len(self.idf))
        num_groups = len(set(self.ground_truth))
        counter = collections.Counter(self.ground_truth)
        results = dict((key,{}) for key in counter)
        info =  dict((key,{}) for key in counter)
        print results
        print counter
        for xpath in self.df:
            print "---- " + xpath + " -----"
            df = self.df[xpath]
            dic = dict((key,0) for key in counter)
            assert len(self.ground_truth) == len(self.pages)
            for index, group in enumerate(self.ground_truth):
                page = self.pages[index]
                if xpath in page.xpaths_list:
                    dic[group] += 1
            for key, value in dic.iteritems():
                info[key][xpath] = str(value) + "\t" + str(df) + "\t" + str(counter[key]) + "\t" + str(length)
                tmp = float(value **2 * length)/ float((counter[key]**2 * df))
                results[key][xpath] = tmp
            #print pages.ground_truth
            print "------------------------"

        for key in counter:
            xpaths = results[key]
            sorted_list = sorted(xpaths.iteritems(), key=lambda x:x[1], reverse=True)
            top = 10
            print "========" + str(key) + "======="
            total_sum = 0.0
            for item in sorted_list:
                total_sum += float(item[1])
            for i in range(top):
                print str(sorted_list[i][0]) + "\t" + str(sorted_list[i][1]/total_sum)
                print info[key][sorted_list[i][0]]
            print "=========================="

    def find_bigram(self):

        df = self.selected_df

        xpaths_dict = {}
        co_dict = {}
        bigram_list = []
        N_pages = len(self.pages)

        for key1 in df.keys():
            xpaths_dict[key1] = {}
            co_dict[key1] = {}
            for key2 in df.keys():
                if key1 != key2:
                    # if key occurs too little, we don't think it is a stop structure
                    xpaths_dict[key1][key2] = float(N_pages**2)/(float(df[key1])*float(df[key2]))
                    co_dict[key1][key2] = 0
                    bigram_list.append([key1,key2])


        # unordered pair
        for p in range(N_pages):
            page = self.pages[p]
            for pair in bigram_list:
                key1, key2 = pair[0], pair[1]
                if page.xpaths[key1] >0 and page.xpaths[key2]>0:
                    co_dict[key1][key2] += 1
                    continue


        pair_dict = {}
        for key1 in xpaths_dict:
            for key2 in xpaths_dict[key1]:
                dis_similarity = self.calc_dis_similarity(key1, key2)
                #print key1, key2, dis_similarity

                p = float(co_dict[key1][key2])/float(N_pages)
                xpaths_dict[key1][key2] = p*xpaths_dict[key1][key2]

                if xpaths_dict[key1][key2] == 0:
                    pair_dict["("+key1+","+key2+")"] = 0
                else:
                    pair_dict["("+key1+","+key2+")"] = math.log(xpaths_dict[key1][key2]) * p
                    #* dis_similarity
        bigram_list = []
        top = 10
        pair_list = sorted(pair_dict.iteritems(),key=lambda x:x[1],reverse=True)
        for i in range(top):
            #print pair_list[i][0] + "\t" + str(pair_list[i][1])
            [path1, path2] = pair_list[i][0].replace("(","").replace(")","").split(",")
            #print str(df[path1]) + "\t" + str(df[path2])
            bigram_list.append([path1,path2])
        print bigram_list
        return bigram_list

    # find the postion of exact match and count "/"
    @staticmethod
    def calc_dis_similarity(p1,p2):

        for i in range(min(len(p1),len(p2))):
            if p1[i] != p2[i]:
                break
        n_shared = p1[:i].count("/")
        n_p1 = p1[i:].count("/")
        n_p2 = p2[i:].count("/")
        dis_similarity = float(n_p1+n_p2+n_shared)/float(n_shared)
        return dis_similarity

if __name__=='__main__':
    #UP_pages = allPages(["../Crawler/crawl_data/Questions/"])
    #pages = allPages(["../Crawler/crawl_data/Questions/"])
    import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument("datasets", choices=["zhihu","stackexchange","rottentomatoes","medhelp","asp"], help="the dataset for experiments")

    pages = allPages(["../Crawler/test_data/rottentomatoes/"])
    df_dict  = pages.df
    sorted_df = sorted(df_dict.iteritems(), key=lambda d:d[0], reverse=False)
    print "we have " + str(len(sorted_df)) + " xpahts in total "
    for item in sorted_df:
        print item[0], item[1]

