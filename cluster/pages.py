import collections
import math
import os
import pickle
from sklearn import metrics
from sets import Set
import operator
import numpy as np
from sklearn.preprocessing import normalize
from url_annotator import  annotator

from cluster import cluster
from page import Page


class allPages:
    def __init__(self, folder_path, dataset, date="Mar15", mode="read"): # mode: {raw, read, write}
        self.folder_path = folder_path
        self.dataset = dataset
        self.date = date
        #print folder_path
        self.threshold = 0.004
        self.pages = []
        self.path_list = []
        self.category = [] # prediction
        self.xpaths_set = Set()
        self.ground_truth = []  # ground truth list for all pages
        self.idf = {}
        self.selected_df = {}
        self.df = {}
        self.features = []
        self.mode = mode
        if not os.path.exists("./{}/feature/".format(date)+dataset):
            os.makedirs("./{}/feature/".format(date)+dataset)
        if mode == "read":
            page_list = open("./{}/feature/".format(date)+dataset+"/pages.txt","r").readlines()
            tf_idf_lines = open("./{}/feature/".format(date)+dataset+"/tf_idf.txt","r").readlines()
            log_tf_idf_lines = open("./{}/feature/".format(date) + dataset + "/log_tf_idf.txt","r").readlines()
            features = open("./{}/feature/".format(date) + dataset + "/xpaths.txt","r").readlines()
            idf_file = open("./{}/feature/".format(date) + dataset  + "/idf.txt","r")

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

            for i in range(len(features)):
                fid =features[i].strip().split(":")[0]
                xpath = features[i].strip().split(":")[1]
                self.features.append(xpath)

            self.idf = pickle.load(idf_file)
            self.category = [0 for i in range(len(page_list))]
            self.get_ground_truth(dataset)
        elif mode == "c_baseline":
            print "it is the baseline of v.crescenzi"
            self.add_page_anchor(folder_path)
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
            self.get_ground_truth(dataset)
            
            if mode=="write":
                print "write mode !"
                xpath_file =  open("./{}/feature/".format(date)+dataset+"/xpaths.txt","w")
                print len(self.pages)
                # filtered xpath :  id xpath
                for page in self.pages:
                    xpath_id = 0
                    for xpath in page.selected_tfidf:
                        xpath_file.write(str(xpath_id)+":"+xpath+"\n")
                        self.features.append(xpath)
                        xpath_id+=1
                    break


                page_file =  open("./{}/feature/".format(date)+dataset + "/pages.txt","w")# id file_path
                tf_idf_file =  open("./{}/feature/".format(date) + dataset + "/tf_idf.txt","w")  # pid features..
                log_tf_idf_file = open("./{}/feature/".format(date) + dataset +"/log_tf_idf.txt","w")
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
                idf_file = open("./{}/feature/".format(date) + dataset + "/idf.txt","w")
                pickle.dump(self.idf,idf_file)

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
                    print file_path
                    file_page = Page(file_path)
                    # the same number for pags & category
                    self.pages.append(file_page)
                    self.path_list.append(file_page.path)
                    self.category.append(category_num)
                    self.update_xpaths_set(file_page)
            category_num+=1


    def add_page_anchor(self,folder_path_list):
        for folder_path in folder_path_list:
            for html_file in os.listdir(folder_path):
                if ".DS_Store" not in html_file:
                    file_path = folder_path + html_file
                    print file_path
                    file_page = Page(file_path,mode="c_baseline")
                    # the same number for pags & category
                    self.pages.append(file_page)
                    self.path_list.append(file_page.path)

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
        #print gold_dict
        return gold_dict

    def get_ground_truth(self,dataset):
        print "our dataset is {0}".format(dataset)

        data = dataset.replace("new_","")
        if os._exists("./crawling/{0}/site.gold/{1}/{1}.gold".format(self.date,data)):
            print "./crawling/{0}/site.gold/{1}/{1}.gold".format(self.date,data)
            gold_file = open("./crawling/{0}/site.gold/{1}/{1}.gold".format(self.date,data)).readlines()
        elif os._exists("./{0}/site.gold/{1}/{1}.gold".format(self.date,data)):
            gold_file = open("./{0}/site.gold/{1}/{1}.gold".format(self.date,data)).readlines()
            print "./{0}/site.gold/{1}/{1}.gold".format(self.date,data)
        else:
            a = annotator(dataset)
            self.ground_truth = a.get_ground_truth(self.path_list)
            return None

        gold_dict = self.build_gold(gold_file)
        #print self.folder_path
        print gold_dict.keys()
        print "length is ", len(gold_dict.keys())
        for i in range(len(self.pages)):
            # here {}/sample instead of {}_samples
            #path = self.pages[i].path.replace("../Crawler/{0}/samples/{1}/".format(self.date,data),"")
            path = self.pages[i].path.replace("../../Crawler/{0}/samples/{1}/".format(self.date,data),"")
            print path.strip()
            id = int(gold_dict[path.strip().replace(" ","")])
            self.ground_truth.append(id)

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


    def Evaluation(self):
        labels_true = self.ground_truth
        labels_pred = self.category

        new_labels_true = []
        new_labels_pred = []
        outlier_list = [0,0,0] # #outlier from true, common, pred
        for idx, val in enumerate(labels_pred):
            if val == -1 and labels_true[idx] == -1:
                outlier_list[1] +=1
            elif val !=-1 and labels_true[idx] == -1:
                #print str(val) + " " + str(labels_true[idx])
                outlier_list[0] += 1
            elif val ==-1 and labels_true[idx] != -1:
                outlier_list[2] += 1
            if val != -1 and labels_true[idx]!= -1:
                new_labels_pred.append(val)
                new_labels_true.append(labels_true[idx])
        for i in range(len(labels_true)):
            print labels_true[i], labels_pred[i]

        #train_batch_file = open("./results/train_batch.results","a")
        prefix =  str(self.dataset)
        #train_batch_file.write(prefix + "#class/#cluster\t" + "{}/{}".format(len(set(labels_true))-1,len(set(labels_pred))-1)+"\n")
        #train_batch_file.write(prefix + "#new_outlier\t" + str(outlier_list[2])+"\n")

        print "number of -1 " + str(len(labels_true)-len(new_labels_true))
        print "we have number of classes from ground truth is {0}".format(len(set(labels_true)))
        print "we have number of classes from clusters is {0}".format(len(set(labels_pred))-1)

        print "Outlier: Cover {1} of {0} total ground truth, and create {2} outlier in prediction. ".format(outlier_list[0]+outlier_list[1],outlier_list[1],outlier_list[2])

        labels_true, labels_pred = new_labels_true, new_labels_pred

        #path_list = self.path_list
        '''
        pred_result_file = open("./clustering/{}_{}_{}.txt".format(dataset,algo,feature),"w")
        for index,label_pred in enumerate(self.UP_pages.category):
            #print path_list[index] + "\t" + str(label_true) + "\t" + str(label_pred)
            pred_result_file.write(path_list[index] + "\tgold: " + str(self.UP_pages.ground_truth[index]) + "\tcluster: " + str(label_pred) + "\n")
        '''
        print "We have %d pages for ground truth!" %(len(labels_true))
        print "We have %d pages after prediction!" %(len(labels_pred))
        assert len(labels_true) == len(labels_pred)
        #self.Precision_Recall_F(labels_true,labels_pred)
        mutual_info_score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
        rand_score = metrics.adjusted_rand_score(labels_true, labels_pred)

        #silhouette_score = metrics.silhouette_score(self.X,np.array(labels_pred), metric='euclidean')
        #print "Silhouette score is " + str(silhouette_score)
        [micro_f, macro_f,micro_p,macro_p] = self.F_Measure(labels_true,labels_pred)
        print "Mutual Info Score is " + str(mutual_info_score)
        print "Adjusted Rand Score is " + str(rand_score)
        print "Micro F-Measure is " + str(micro_f)
        print "Macro F-Measure is " + str(macro_f)
        print "Micro Precision is " + str(micro_p)
        print "Macro Precision is " + str(macro_p)



        #train_batch_file.write("=====" + str(dataset) + "\t" + str(algo) +  "\t" + str(feature) +  "=====\n")
        metrics_list = ['micro_f', 'macro_f', 'micro_p', 'macro_p']
        result = [micro_f,macro_f,micro_p,macro_p]
        #for index,metric in enumerate(metrics_list):
            #line =  prefix + metric + "\t" + str(result[index])
            #train_batch_file.write(line + "\n" )
        return micro_f, macro_f, micro_p, macro_p, mutual_info_score, rand_score

    def F_Measure(self,labels_true,labels_pred):
        ground_truth_set = set(labels_true)
        pre_set = set(labels_pred)
        # dict with index and cluster_index:
        length = len(labels_true)
        ng = {}
        np = {}
        precision = {}
        recall = {}
        fscore = {}
        labels = {}
        # final return
        micro_f1,micro_p = 0.0,0.0
        macro_f1,macro_p = 0.0,0.0
        for item in ground_truth_set:
            labels[item] = {}
            precision[item] = {}
            recall[item] = {}
            fscore[item] = {}
            for item2 in pre_set:
                labels[item][item2] = 0

        # get the distribution of clustering results
        for i in range(length):
            g_index = labels_true[i]
            p_index = labels_pred[i]
            labels[g_index][p_index] += 1
            if ng.has_key(g_index):  # number of ground truth
                ng[g_index] += 1
            else:
                ng[g_index] = 1
            if np.has_key(p_index):
                np[p_index] += 1
            else:
                np[p_index] = 1
        # get the statistical results
        for i in ground_truth_set:
            for j in pre_set:
                if np[j]==0:
                    print str(j) + " is zero"
                recall[i][j] = float(labels[i][j])/float(ng[i])
                precision[i][j] = float(labels[i][j])/float(np[j])
                if recall[i][j]*precision[i][j]==0:
                    fscore[i][j] = 0.0
                else:
                    fscore[i][j] = (2*recall[i][j]*precision[i][j])/(recall[i][j]+precision[i][j])

        for i in ground_truth_set:
            tmp_max = max(fscore[i].iteritems(), key=operator.itemgetter(1))[1]
            micro_f1 += tmp_max*ng[i]/float(length)
            macro_f1 += tmp_max/float(len(ground_truth_set))
            #micro_f1 += tmp_max/float(len(ground_truth_set))


        ## flawed !!!1
        cluster_dict = self.get_cluster_number_shift(labels_true, labels_pred)
        right_guess = 0
        test_gold_counter = collections.Counter(labels_true)
        test_gold_right = dict([(index,0.0) for index in test_gold_counter])
        for index,item in enumerate(labels_pred):
            if cluster_dict[item] == labels_true[index]:
                test_gold_right[labels_true[index]] += 1
                right_guess += 1

        micro_p = float(right_guess)/float(len(labels_true))
        avg = 0.0
        for index in test_gold_counter:
            avg += float(test_gold_right[index])/float(test_gold_counter[index])
        macro_p = avg/float(len(test_gold_counter))

        return [micro_f1,macro_f1,micro_p,macro_p]

    def get_cluster_number_shift(self,labels_true, labels_pred):
        true_set = set(labels_true)
        pre_set = set(labels_pred)
        #print pre_set
        dic = {}
        for item in pre_set:
            dic[item] = {}
            for item_2 in true_set:
                dic[item][item_2] = 0
        assert len(labels_true) == len(labels_pred)

        for i in range(len(labels_true)):
            dic[labels_pred[i]][labels_true[i]] += 1
        print "ground truth data"

        #print dic
        self.output_dict(dic)

        final_dict = collections.defaultdict(dict)
        #used_list = set()
        for pred_key in pre_set:
            max_value = -1
            #print dic[pred_key]
            for index, value in dic[pred_key].iteritems():
                #if index not in used_list:
                if value > max_value:
                    max_label = index
                    max_value = value
                final_dict[pred_key] = max_label
                #used_list.add(max_label)
        return final_dict

    def output_dict(self,dic):
        for key in dic:
            print "cluster No. is " + str(key) + " ->{ ",
            value_dic = dic[key]
            for class_key in value_dic:
                if value_dic[class_key]!=0:
                    print "'"+str(class_key)+"'" + ": " + str(value_dic[class_key]) + ", ",
            print " }"




if __name__=='__main__':
    #UP_pages = allPages(["../Crawler/crawl_data/Questions/"])
    #pages = allPages(["../Crawler/crawl_data/Questions/"])
    #parser = argparse.ArgumentParser()
    #parser.add_argument("datasets", choices=["zhihu","stackexchange","rottentomatoes","medhelp","asp"], help="the dataset for experiments")

    pages = allPages(["../Crawler/test_data/rottentomatoes/"],"rotten",mode="c_baseline")
    for page in pages.pages:
        print page.path, page.anchor_xpath_set


