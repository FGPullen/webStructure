from pages import allPages
import sys
from wkmeans import WKMeans
from kmeans import KMeans
import numpy as np
import scipy as sp
#import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn.cluster as Cluster
#from visualization import visualizer
from sklearn.utils import shuffle
from sklearn.preprocessing import scale,normalize
from time import time
import operator
from sklearn.cross_validation import StratifiedKFold
from cluster import cluster
import collections
from sets import Set
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier

class pagesCluster:


    def __init__(self, dataset, path_list=None,num_clusters=None):
        if path_list is None and num_clusters is None:
            self.t0 = time()
        else:
            self.t0 = time()
            self.dataset = dataset
            self.UP_pages = allPages(path_list,dataset)
            self.path_list = self.UP_pages.path_list
            self.num_clusters = num_clusters
        #self.clustering(num_clusters)

    def wkmeans(self,num_clusters,features, weight_method=None, cv=False, beta=2, replicates=100):
        feature_matrix = []
        y =[]
        # get features and labels
        time = 1

        tf_feat = open("./results/medhelp-tf-idf.csv","w")
        b_feat = open("./results/medhelp-binary.csv","w")
        for page in self.UP_pages.pages:
            tf_feat.write(page.path)
            b_feat.write(page.path)
            for key in page.selected_tfidf:
                tf_feat.write("\t" + str(page.selected_tfidf[key]))
            for key in page.Leung:
                b_feat.write("\t" + str(page.Leung[key]))
            tf_feat.write("\n")
            b_feat.write("\n")

        df = self.UP_pages.df
        for key in df:
            print key + "\t" + str(df[key])


        for page in self.UP_pages.pages:
            if features == "tf-idf":
                vector = []
                for key in page.selected_tfidf:
                    vector.append(page.selected_tfidf[key])
                vector = normalize(vector,norm='l1')[0]
                feature_matrix.append(vector)

            elif features == "log-tf-idf":
                vector = []
                for key in page.selected_logtfidf:
                    vector.append(page.selected_logtfidf[key])
                vector = normalize(vector,norm='l1')[0]
                feature_matrix.append(vector)

            elif features == "binary":
                vector = []
                for key in page.Leung:
                    vector.append(page.Leung[key])
                vector = normalize(vector,norm='l1')[0]
                feature_matrix.append(vector)

        self.X = np.array(feature_matrix)

        if not cv:
            print "the size of vector is " + str(self.X.shape)
            t = WKMeans()
            final_u, final_centroids, weights, final_ite, final_dist = t.wk_means(self.X,num_clusters,beta=beta,replicates=replicates, weight_method=weight_method)
            self.pre_y = final_u
            self.UP_pages.updateCategory(self.pre_y)
            print "we have avg interation for " + str(final_ite)

            #write_file = open("./Files/values.txt","w")

            keys = []
            for key in page.selected_tfidf:
                keys.append(key)
            return t
            '''
            for group in weights:
                print "-------"
                sorted_list= sorted(enumerate(group), key=lambda d:d[1], reverse = False)
                for i in range(200):
                    key = sorted_list[i][0]
                    value = sorted_list[i][1]
                    if '/a' in keys[key]:
                        print str(keys[key]) + "\t" + str(value)
                print "-------"
            print self.pre_y
            '''
        else :
            labels_true = np.array(self.UP_pages.ground_truth)
            skf = StratifiedKFold(labels_true, n_folds=4)
            results = []
            for train, test in skf:
                train_x, test_x,train_gold,test_gold = self.X[train], self.X[test], labels_true[train], labels_true[test]
                t = WKMeans()
                train_y, final_centroids, weights, final_ite, final_dist = t.wk_means(train_x,num_clusters,beta=beta,replicates=replicates, weight_method=weight_method)
                test_y = t.wk_means_classify(test_x)
                results.append(self.Evaluation_CV(test_gold,test_y,train_gold,train_y))
                #results.append(self.Cv_Evaluation(test_gold,test_y))
            result = np.mean(results,axis=0)
            cv_batch_file = open("./results/cv_batch.results","a")
            cv_batch_file.write("=====" + str(self.UP_pages.folder_path[0]) +"\t" + features + "\twkmeans =====\n")
            metrics = ['micro_f', 'macro_f', 'mutual_info_score', 'rand_score', 'cv_micro_precision','cv_macro_precision']
            for index,metric in enumerate(metrics):
                print metric + "\t" + str(result[index])
                cv_batch_file.write(metric + "\t" + str(result[index])+"\n")


    def kmeans(self,num_clusters,features,cv=False,replicates=100):
        feature_matrix = []
        y =[]
        # get features and labels

        for page in self.UP_pages.pages:
            # selected normalized tf idf
            if features == "tf-idf":
                vector = []
                for key in page.selected_tfidf:
                    vector.append(page.selected_tfidf[key])
                #for key,value in page.bigram_dict.iteritems():
                #    vector.append(value)
                #vector = normalize(vector,norm='l1')[0]
                feature_matrix.append(vector)

            elif features == "log-tf-idf":
                vector = []
                for key in page.selected_tfidf:
                    vector.append(page.selected_logtfidf[key])
                #for key,value in page.bigram_dict.iteritems():
                #    vector.append(value)
                #vector = normalize(vector,norm='l1')[0]
                feature_matrix.append(vector)

            elif features == "binary":
                vector = []
                for key in page.Leung:
                    vector.append(page.Leung[key])
                #for key,value in page.bigram_dict.iteritems():
                #    vector.append(value)
                #vector = normalize(vector,norm='l1')[0]
                feature_matrix.append(vector)

        self.X = normalize(np.array(feature_matrix), norm='l1')
        #print self.X
        print "the size of vector is " + str(self.X.shape)

        #self.X = scale(self.X)
        # select
        #num_clusters = len(path_list)
        if not cv:
            print "number of clusters is " + str(num_clusters)
            '''
            k_means = Cluster.KMeans(n_clusters=num_clusters, n_init=100, random_state=1, n_jobs=2)
            #k_means.fit(shuffle(self.X))
            #self.pre_y = k_means.predict(self.X)
            k_means.fit(self.X)
            self.pre_y = k_means.labels_
            self.UP_pages.updateCategory(self.pre_y)
            '''

            print "the size of vector is " + str(self.X.shape)
            t = KMeans()
            final_u, final_centroids, final_ite, final_dist = t.k_means(self.X,num_clusters, replicates=replicates)
            self.pre_y = final_u
            print self.pre_y
            self.UP_pages.updateCategory(self.pre_y)
            print "we have avg interation for " + str(final_ite)
            keys = []
            for key in page.selected_tfidf:
                keys.append(key)
            return t

        else:
            labels_true = np.array(self.UP_pages.ground_truth)
            skf = StratifiedKFold(labels_true, n_folds=5)
            results = []
            for train, test in skf:
                #print train, test
                train_x, test_x,train_gold,test_gold = self.X[train], self.X[test], labels_true[train], labels_true[test]
                t = KMeans()
                train_y, final_centroids, final_ite, final_dist = t.k_means(train_x,num_clusters,replicates=replicates)
                test_y = t.k_means_classify(test_x)
                path_list = self.path_list
                results.append(self.Evaluation_CV(test_gold,test_y,train_gold,train_y, path_list=path_list))
            result = np.mean(results,axis=0)

            cv_batch_file = open("./results/cv_batch.results","a")
            algo = "kmeans"
            dataset = self.dataset
            prefix =  str(dataset) + "\t" + str(algo) + "\t" + str(features) + "\t"
            metrics = ['cv_micro_precision','cv_macro_precision']
            for index,metric in enumerate(metrics):
                line =  prefix + metric + "\t" + str(result[index])
                cv_batch_file.write(line + "\n" )


    def AgglomerativeClustering(self, num_clusters):

        #self.X = np.array([[0,1,2,4],[1,0,3,4],[2,3,0,1],[4,4,1,0]])
        #self.X = self.get_affinity_matrix()
        #self.X = self.get_edit_distance_matrix()
        #self.X = self.read_edit_distance_matrix()
        self.X = self.UP_pages.get_one_hot_distance_matrix()
        print "start?"
        ahc = Cluster.AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed',linkage='complete')
        ahc.fit(self.X)
        self.pre_y = ahc.labels_
        self.UP_pages.updateCategory(self.pre_y)
        print self.pre_y

    def findEps(self, X):
        K = 4
        bin_num = 100
        default_eps = 0.2
        kdist_list = []
        nbrs = NearestNeighbors(n_neighbors=K, algorithm="ball_tree").fit(X)
        distances, indices = nbrs.kneighbors(X)
        for dist in distances:
            kdist_list += dist.tolist()[1:]
        n, bins = np.histogram(kdist_list, bins=bin_num)
        threshold = np.mean(n[bin_num/3:])
        eps = default_eps
        for idx, val in enumerate(n):
            if idx > 5 and val < threshold:
                eps = bins[idx]
                break
        return eps

    def DBSCAN(self, features, cv=False):
        print "The feature is " + str(features) + " with DBSCAN"
        # default file path is
        feature_matrix = []
        y = []

        for page in self.UP_pages.pages:
            if features == "tf-idf":
                vector = []
                for key in page.selected_tfidf:
                    vector.append(page.selected_tfidf[key])
                feature_matrix.append(vector)
            elif features == "log-tf-idf":
                vector = []
                for key in page.selected_tfidf:
                    vector.append(page.selected_logtfidf[key])
                feature_matrix.append(vector)

            elif features == "binary":
                vector = []
                for key in page.Leung:
                    vector.append(page.Leung[key])
                feature_matrix.append(vector)

        self.X = normalize(np.array(feature_matrix), norm='l1')
        print "the size of vector is " + str(self.X.shape)

        if cv:
            labels_true = np.array(self.UP_pages.ground_truth)
            skf = StratifiedKFold(labels_true, n_folds=4)
            results = []
            for train, test in skf:
                #print train, test
                train_x, test_x, train_gold, test_gold = self.X[train], self.X[test], labels_true[train], labels_true[test]
                eps = self.findEps(train_x)
                print "eps is " + str(eps)
                db = Cluster.DBSCAN(eps=eps, min_samples=4).fit(train_x)
                train_y = db.labels_
                #self.get_cluster_number_shift(train_gold, train_y)

                # determine the number of clusters by DBSCAN
                num_clusters = len(set(train_y)) - 1
                '''
                km_train_x = []
                km_train_gold = []
                for idx, val in enumerate(train_y):
                    if val != -1:
                        km_train_x.append(train_x[idx])
                        km_train_gold.append(train_gold[idx])
                train_x = np.array(km_train_x)
                train_gold = np.array(km_train_gold)
                '''
                #km_train_x = train_x
                #km_train_gold = train_gold
                K = 3
                nbrs = KNeighborsClassifier(n_neighbors=K,weights="distance",algorithm="ball_tree").fit(train_x,train_y)
                test_y = nbrs.predict(test_x)
                path_list = self.UP_pages.path_list
                #for i in [test_gold,test_y,train_gold,train_y]:
                #    print len(i)
                results.append(self.Evaluation_CV(test_gold,test_y, train_gold, train_y, path_list=path_list))

                '''
                t = KMeans()
                train_y, final_centroids, final_ite, final_dist = t.k_means(km_train_x, num_clusters, replicates=20)
                test_y = t.k_means_classify(test_x)
                path_list = [self.UP_pages.path_list[idx] for idx in test]
                results.append(self.Evaluation_CV(test_gold,test_y,km_train_gold,train_y, path_list=path_list))
                '''
            result = np.mean(results,axis=0)
            cv_batch_file = open("./results/cv_batch.results","a")
            algo = "dbscan"
            dataset = self.dataset
            prefix =  str(dataset) + "\t" + str(algo) + "\t" + str(features) + "\t"
            metrics = ['cv_micro_precision','cv_macro_precision']
            for index,metric in enumerate(metrics):
                line =  prefix + metric + "\t" + str(result[index])
                print line
                cv_batch_file.write(line + "\n" )

        else:
            print "the size of vector is " + str(self.X.shape)
            eps = self.findEps(self.X)
            #print "eps is " + str(eps)
            db = Cluster.DBSCAN(eps=eps, min_samples=4).fit(self.X)
            self.pre_y = db.labels_
            self.UP_pages.updateCategory(self.pre_y)
            num_clusters = len(set(self.pre_y)) - 1
            #print "number of dbscan cluster is " + str(num_clusters)
            '''
            write_file = open(self.dataset+".txt","w")
            for i in xrange(len(self.UP_pages.pages)):
                write_file.write(self.UP_pages.pages[i].path + " gold:" + str(self.UP_pages.ground_truth[i]) \
                                + " cluster: " + str(self.UP_pages.category[i]))
            '''

    def get_affinity_matrix(self):
        return self.UP_pages.get_affinity_matrix()

    def get_edit_distance_matrix(self):
        return self.UP_pages.get_edit_distance_matrix()

    def read_edit_distance_matrix(self):
        return self.UP_pages.read_edit_distance_matrix()

    def Output(self):
        write_file = open("cluster_result.txt","w")
        assert len(self.pre_y) == len(self.UP_pages.pages)
        for i in range(len(self.pre_y)):
            tmp = self.filename2Url(self.UP_pages.pages[i].path) + "\t" + str(self.pre_y[i])
            write_file.write(tmp + "\n")

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




    def Evaluation_CV(self, test_gold, test_y, train_gold, train_y, path_list=None):
        if test_gold is None or test_y is None or train_gold is None or train_y is None:
            raise "Labels are None"

        new_test_gold = []
        new_test_y = []
        for idx, val in enumerate(test_y):
            if val != -1 and test_gold[idx]!=-1:
                new_test_y.append(val)
                new_test_gold.append(test_gold[idx])

        print "number of -1 " + str(len(new_test_gold)-len(test_gold))
        test_gold, test_y = new_test_gold, new_test_y


        #test_gold,test_y,train_gold,train_y
        test_gold_counter = collections.Counter(test_gold)
        test_gold_right = dict([(index,0.0) for index in test_gold_counter])
        cluster_dict = self.get_cluster_number_shift(train_gold, train_y) # cluster_dict: key: cluster_id value: class_id
        #print cluster_dict
        right_guess = 0
        for index,item in enumerate(test_y):
            if cluster_dict[item] == test_gold[index]:
                test_gold_right[test_gold[index]] += 1
                right_guess += 1
            else:
                # wrong classification here
                if path_list is not None:
                    path = path_list[index]
                    cluster_id = cluster_dict[item]
                    print "Wrong instance:\nPage: {0}\nResult Cluster={1}, True Cluster={2}".format(path, cluster_id, test_gold[index])
        micro_precision = float(right_guess)/float(len(test_y))
        print "===examine==="
        print test_gold_counter
        print test_gold_right
        print test_y

        avg = 0.0
        for index in test_gold_counter:
            avg += float(test_gold_right[index])/float(test_gold_counter[index])
        macro_precision = avg/float(len(test_gold_counter))

        print "We have %d pages for ground truth!" %(len(train_y))
        print "We have %d pages after prediction!" %(len(test_y))
        assert len(test_gold) == len(test_y)
        assert len(train_gold) == len(train_y)
        pages = self.UP_pages
        #self.Precision_Recall_F(labels_true,labels_pred)
        mutual_info_score = metrics.adjusted_mutual_info_score(test_gold, test_y)
        rand_score = metrics.adjusted_rand_score(test_gold, test_y)
        print "Mutual Info Score is " + str(mutual_info_score)
        print "Adjusted Rand Score is " + str(rand_score)
        #silhouette_score = metrics.silhouette_score(self.X,np.array(labels_pred), metric='euclidean')
        #print "Silhouette score is " + str(silhouette_score)
        [micro_f, macro_f, micro_p, macro_p] = self.F_Measure(test_gold,test_y)
        # here the micro_p and macro_p is useless
        print "Micro F-Measure is " + str(micro_f)
        print "Macro F-Measure is " + str(macro_f)
        print "Micro CV precision is " + str(micro_precision)
        print "Macro CV precision is " + str(macro_precision)
        return micro_precision, macro_precision, micro_f, macro_f, mutual_info_score, rand_score

    def Evaluation(self,dataset,algo,feature):
        labels_true = self.UP_pages.ground_truth
        labels_pred = self.UP_pages.category

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

        train_batch_file = open("./results/train_batch.results","a")
        prefix =  str(dataset) + "\t" + str(algo) + "\t" + str(feature) + "\t"
        train_batch_file.write(prefix + "#class/#cluster\t" + "{}/{}".format(len(Set(labels_true))-1,len(Set(labels_pred))-1)+"\n")
        train_batch_file.write(prefix + "#new_outlier\t" + str(outlier_list[2])+"\n")

        #print "number of -1 " + str(len(labels_true)-len(new_labels_true))
        print "we have number of classes from ground truth is {0}".format(len(Set(labels_true)))
        print "we have number of classes from clusters is {0}".format(len(Set(labels_pred))-1)

        print "Outlier: Cover {1} of {0} total ground truth, and create {2} outlier in prediction. ".format(outlier_list[0]+outlier_list[1],outlier_list[1],outlier_list[2])

        labels_true, labels_pred = new_labels_true, new_labels_pred

        path_list = self.path_list

        pred_result_file = open("./clustering/{}_{}_{}.txt".format(dataset,algo,feature),"w")
        for index,label_pred in enumerate(self.UP_pages.category):
            #print path_list[index] + "\t" + str(label_true) + "\t" + str(label_pred)
            pred_result_file.write(path_list[index] + "\tgold: " + str(self.UP_pages.ground_truth[index]) + "\tcluster: " + str(label_pred) + "\n")

        print "We have %d pages for ground truth!" %(len(labels_true))
        print "We have %d pages after prediction!" %(len(labels_pred))
        assert len(labels_true) == len(labels_pred)
        pages = self.UP_pages
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
        for index,metric in enumerate(metrics_list):
            line =  prefix + metric + "\t" + str(result[index])
            train_batch_file.write(line + "\n" )
        return micro_f, macro_f, micro_p, macro_p, mutual_info_score, rand_score


    def filename2Url(self,filename):
        return filename.replace("_","/")

    def get_top_local_xpath(self,threshold, group):
        pages = self.UP_pages
        global_threshold = len(pages.pages) * threshold
        assert len(pages.pages) == len(cluster_labels.pre_y)
        for i in range(len(cluster_labels.pre_y)):
            if group in pages.pages[i].path:
                users_num[cluster_labels.pre_y[i]] += 1
            clusters[cluster_labels.pre_y[i]].addPage(pages.pages[i])
        index, value = max(enumerate(users_num), key=operator.itemgetter(1))
        #print str(index) + "\t" + str(value)
        user_cluster = clusters[index]
        user_cluster.find_local_stop_structure(pages.nidf,global_threshold)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", choices=["zhihu","stackexchange","rottentomatoes","medhelp","asp","new_stackexchange",\
                        "new_rottentomatoes","new_asp","new_douban","new_youtube","new_tripadvisor","new_hupu","new_baidu","new_amazon","new_biketo"], help="the dataset for experiments")
    parser.add_argument("clustering", choices=["wkmeans","kmeans","ahc", "dbscan"], help="the algorithm for clustering")
    parser.add_argument("features_type", choices=["tf-idf","log-tf-idf","binary"], help="the features type for clustering")
    parser.add_argument("test_type", choices=["train","cv"], help="clustering or cv?")
    #parser.add_argument('-w', action='store_true')
    # representation option for args
    args = parser.parse_args()
    features_type = args.features_type
    if args.datasets == "zhihu":
        num_clusters = 4
        #cluster_labels = pagesCluster(["../Crawler/crawl_data/Zhihu/"],num_clusters)
        cluster_labels = pagesCluster(args.datasets,["../Crawler/test_data/zhihu/"],num_clusters)
    elif args.datasets == "stackexchange":
        num_clusters = 5
        #cluster_labels = pagesCluster(["../Crawler/crawl_data/Questions/"],num_clusters)
        cluster_labels = pagesCluster(args.datasets,["../Crawler/test_data/stackexchange/"],num_clusters)
    elif args.datasets == "test":
        num_clusters = 5
        cluster_labels = pagesCluster(args.datasets,["../Crawler/test_data/train/"],num_clusters)
    elif args.datasets == "rottentomatoes":
        num_clusters = 7
        cluster_labels = pagesCluster(args.datasets,["../Crawler/test_data/rottentomatoes/"],num_clusters)
    elif args.datasets == "medhelp":
        num_clusters = 4
        cluster_labels = pagesCluster(args.datasets,["../Crawler/test_data/medhelp/"],num_clusters)
    elif args.datasets == "asp":
        num_clusters = 6
        cluster_labels = pagesCluster(args.datasets,["../Crawler/test_data/ASP/"],num_clusters)
    elif args.datasets == "new_stackexchange":
        num_clusters = 12
        cluster_labels = pagesCluster(args.datasets,["../Crawler/Mar15_samples/stackexchange/"],num_clusters)
    elif args.datasets == "new_rottentomatoes":
        num_clusters = 16
        cluster_labels = pagesCluster(args.datasets,["../Crawler/Mar15_samples/rottentomatoes/"],num_clusters)
    elif args.datasets == "new_asp":
        num_clusters = 10
        cluster_labels = pagesCluster(args.datasets,["../Crawler/Mar15_samples/asp/"],num_clusters)
    elif args.datasets == "new_douban":
        num_clusters = 8
        cluster_labels = pagesCluster(args.datasets, ["../Crawler/Mar15_samples/douban/"], num_clusters)
    elif args.datasets == "new_youtube":
        num_clusters = 4
        cluster_labels = pagesCluster(args.datasets, ["../Crawler/Mar15_samples/youtube/"], num_clusters)
    elif args.datasets == "new_tripadvisor":
        num_clusters = 18
        cluster_labels = pagesCluster(args.datasets,["../Crawler/Mar15_samples/tripadvisor/"], num_clusters)
    elif args.datasets == "new_hupu":
        num_clusters = 3
        cluster_labels = pagesCluster(args.datasets,["../Crawler/Mar15_samples/hupu/"], num_clusters)
    elif args.datasets == "new_baidu":
        num_clusters = 12
        cluster_labels = pagesCluster(args.datasets, ["../Crawler/Mar15_samples/baidu/"], num_clusters)
    elif args.datasets == "new_amazon":
        num_clusters = 10;
        cluster_labels = pagesCluster(args.datasets, ["../Crawler/Mar15_samples/amazon/"], num_clusters)
    elif args.datasets == "new_biketo":
        num_clusters = 10;
        cluster_labels = pagesCluster(args.datasets, ["../Crawler/Mar15_samples/biketo/"],num_clusters)
    else:
        print "error"


    if args.clustering == "kmeans":
        if args.test_type == "cv":
            cluster_labels.kmeans(cluster_labels.num_clusters,features_type,cv=True,replicates=20)
        else:
            cluster_labels.kmeans(cluster_labels.num_clusters,features_type,replicates=100)
            cluster_labels.Evaluation(args.datasets,args.clustering,features_type)

    elif args.clustering == "wkmeans":
        if args.test_type == "cv":
            cluster_labels.wkmeans(cluster_labels.num_clusters,features_type,cv=True,beta=2,replicates=20)
        else:
            cluster_labels.wkmeans(cluster_labels.num_clusters,features_type,replicates=100)
            cluster_labels.Evaluation(args.datasets,args.clustering,features_type)
    #elif arg.clustering == "all":

    elif args.clustering == "ahc":
        cluster_labels.AgglomerativeClustering(cluster_labels.num_clusters, replicates=100)
        cluster_labels.Evaluation()

    elif args.clustering == 'dbscan':
        if args.test_type == "cv":
            cluster_labels.DBSCAN(features_type,  cv=True)
        else:
            cluster_labels.DBSCAN(features_type, cv=False)
            cluster_labels.Evaluation(args.datasets,args.clustering,features_type)
    #visualization
    '''
    if args.test_type != "cv":
        v = visualizer(cluster_labels.UP_pages)
        twoD_file = "2D_plot_file.txt"
        v.show(v.UP_pages.ground_truth, v.UP_pages.category ,twoD_file, args.datasets)
    '''

