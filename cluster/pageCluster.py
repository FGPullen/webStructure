from pages import allPages
import sys
from wkmeans import WKMeans
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn.cluster as Cluster
from gensim.models import word2vec
from visualization import visualizer
from sklearn.utils import shuffle
from sklearn.preprocessing import scale,normalize
from time import time
import operator
from sklearn.cross_validation import StratifiedKFold
from cluster import cluster
import collections

class pagesCluster:


	def __init__(self, path_list=None,num_clusters=None):
		if path_list is None and num_clusters is None:
			self.t0 = time()
		else:
			self.t0 = time()
			self.UP_pages = allPages(path_list)
			self.num_clusters = num_clusters
		#self.clustering(num_clusters)
		
	def wkmeans(self,num_clusters,features, weight_method=None, cv=False, beta=2, replicates=100):
		feature_matrix = []
		y =[]
		# get features and labels
		time = 1
		for page in self.UP_pages.pages:
			if features == "tf-idf":
				vector = []
				for key in page.selected_tfidf:
					vector.append(page.selected_tfidf[key])
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

			write_file = open("./Files/values.txt","w")

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
			skf = StratifiedKFold(labels_true, n_folds=5)
			results = []
			for train, test in skf:
				#print train, test
				train_x, test_x,train_gold,test_gold = self.X[train], self.X[test], labels_true[train], labels_true[test]
				t = WKMeans()
				train_y, final_centroids, weights, final_ite, final_dist = t.wk_means(train_x,num_clusters,beta=beta,replicates=replicates, weight_method=weight_method)
				test_y = t.wk_means_classify(test_x)
				results.append(self.Evaluation_CV(test_gold,test_y,train_gold,train_y))
				#results.append(self.Cv_Evaluation(test_gold,test_y))
			result = np.mean(results,axis=0)
			metrics = ['micro_f', 'macro_f', 'mutual_info_score', 'rand_score', 'cv_precision']
			for index,metric in enumerate(metrics):
				print metric + "\t" + str(result[index])

	def kmeans(self,num_clusters,features,cv=False):
		feature_matrix = []
		y =[]
		# get features and labels
		
		for page in self.UP_pages.pages:
			# selected normalized tf idf 
			if features == "tf-idf":
				vector = []
				for key in page.selected_tfidf:
					vector.append(page.selected_tfidf[key])
				vector = normalize(vector,norm='l1')[0]
				feature_matrix.append(vector)
			elif features == "binary":
				vector = []
				for key in page.Leung:
					vector.append(page.Leung[key])
				vector = normalize(vector,norm='l1')[0]
				feature_matrix.append(vector)

		self.X = np.array(feature_matrix)
		print "the size of vector is " + str(self.X.shape)
	
		#self.X = scale(self.X)
		# select 
		#num_clusters = len(path_list)
		if not cv:
			print "number of clusters is " + str(num_clusters)
			k_means = Cluster.KMeans(n_clusters=num_clusters, n_init=100, random_state=1, n_jobs=2)
			#k_means.fit(shuffle(self.X))
			#self.pre_y = k_means.predict(self.X)
			k_means.fit(self.X)
			self.pre_y = k_means.labels_
			self.UP_pages.updateCategory(self.pre_y)
			#print self.pre_y
			#print self.y
			#print metrics.adjusted_mutual_info_score(self.UP_pages.ground_truth,self.UP_pages.category)  
			#print("done in %0.3fs." % (time() - self.t0))
		else:
			labels_true = np.array(self.UP_pages.ground_truth)
			skf = StratifiedKFold(labels_true, n_folds=5)
			results = []
			for train, test in skf:
				#print train, test
				train_x, test_x,train_gold,test_gold = self.X[train], self.X[test], labels_true[train], labels_true[test]
				k_means = Cluster.KMeans(n_clusters=num_clusters, n_init=100, random_state=1, n_jobs=2)
				k_means.fit(train_x)
				train_y = k_means.labels_
				test_y = k_means.predict(test_x)
				results.append(self.Evaluation(test_gold,test_y,train_gold,train_y))
			result = np.mean(results,axis=0)
			metrics = ['micro_f', 'macro_f', 'mutual_info_score', 'rand_score','precision']
			for index,metric in enumerate(metrics):
				print metric + "\t" + str(result[index])			

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

	def DBSCAN(self):
		# default file path is 
		feature_matrix = []
		lines = open("./Data/edit_distance.matrix","r").readlines()
		for i in range(len(lines)):
			line = lines[i]
			dis = line.strip().split("\t")
			# normalize 
			for item in range(len(dis)):
				dis[item] = int(dis[item])
			#dis_sum = sum(dis)
			#for item in range(len(dis)):
			#	dis[item] = float(dis[item])/float(dis_sum)

			feature_matrix.append(dis)

		print "We have " + str(len(feature_matrix)) + " pages."
		D = np.array(feature_matrix)
		print D.shape
		db = Cluster.DBSCAN(eps=0.3, metric='precomputed',min_samples=10).fit(D)
		labels = db.labels_
		n_clusters_ = len(set(labels))
		print db.labels_
		print('Estimated number of clusters: %d' % n_clusters_)

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

	@staticmethod
	def F_Measure(labels_true,labels_pred):
		ground_truth_set = set(labels_true)
		labels_set = set(labels_pred)
		# dict with index and cluster_index:
		length = len(labels_true)
		ng = {}
		nc = {}
		precision = {}
		recall = {}
		fscore = {}
		labels = {}
		# final return 
		weighted_f1 = 0.0
		macro_f1 = 0.0
		for item in ground_truth_set:
			labels[item] = {}
			precision[item] = {}
			recall[item] = {}
			fscore[item] = {}
			for item2 in labels_set:
				labels[item][item2] = 0

		# get the distribution of clustering results
		for i in range(length):
			g_index = labels_true[i]
			c_index = labels_pred[i]
			labels[g_index][c_index] += 1
			if ng.has_key(g_index):
				ng[g_index] += 1
			else:
				ng[g_index] = 1
			if nc.has_key(c_index):
				nc[c_index] += 1
			else:
				nc[c_index] = 1
		# get the statistical results
		for i in ground_truth_set:
			for j in labels_set:
				if nc[j]==0:
					print str(j) + " is zero"
				recall[i][j] = float(labels[i][j])/float(ng[i])
				precision[i][j] = float(labels[i][j])/float(nc[j])
				if recall[i][j]*precision[i][j]==0:
					fscore[i][j] = 0.0
				else:
					fscore[i][j] = (2*recall[i][j]*precision[i][j])/(recall[i][j]+precision[i][j])

		for i in ground_truth_set:
			tmp_max = max(fscore[i].iteritems(), key=operator.itemgetter(1))[1]
			weighted_f1 += tmp_max*ng[i]/float(length)
			macro_f1 += tmp_max/float(len(ground_truth_set))
			#weighted_f1 += tmp_max/float(len(ground_truth_set))

		return [weighted_f1,macro_f1]


	def Evaluation_CV(self, true=None, pred=None, train_true=None, train_pred=None):
		if true is None:
			labels_true = self.UP_pages.ground_truth
		else:
			labels_train_true = train_true
			labels_true = true
		if pred is None:
			labels_pred = self.UP_pages.category
		else:
			labels_train_pred = train_pred
			labels_pred = pred
		cluster_dict = self.get_cluster_number_shift(labels_train_true, labels_train_pred)
		print cluster_dict
		right_guess = 0
		for index,item in enumerate(pred):
			if cluster_dict[item] == true[index]:
				right_guess += 1
		precision = float(right_guess)/float(len(true))

		print "We have %d pages for ground truth!" %(len(labels_true))
		print "We have %d pages after prediction!" %(len(labels_pred))
		assert len(labels_true) == len(labels_pred)
		pages = self.UP_pages
		#self.Precision_Recall_F(labels_true,labels_pred)
		mutual_info_score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
		rand_score = metrics.adjusted_rand_score(labels_true, labels_pred)
		print "Mutual Info Score is " + str(mutual_info_score)
		print "Adjusted Rand Score is " + str(rand_score)
		#silhouette_score = metrics.silhouette_score(self.X,np.array(labels_pred), metric='euclidean')
		#print "Silhouette score is " + str(silhouette_score)
		[micro_f, macro_f] = self.F_Measure(labels_true,labels_pred)
		print "Micro F-Measure is " + str(micro_f)
		print "Macro F-Measure is " + str(macro_f)
		print "CV precision is " + str(precision)
		return micro_f, macro_f, mutual_info_score, rand_score, precision

	def Evaluation(self):
		labels_true = self.UP_pages.ground_truth
		labels_pred = self.UP_pages.category


		print "We have %d pages for ground truth!" %(len(labels_true))
		print "We have %d pages after prediction!" %(len(labels_pred))
		assert len(labels_true) == len(labels_pred)
		pages = self.UP_pages
		#self.Precision_Recall_F(labels_true,labels_pred)
		mutual_info_score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
		rand_score = metrics.adjusted_rand_score(labels_true, labels_pred)
		print "Mutual Info Score is " + str(mutual_info_score)
		print "Adjusted Rand Score is " + str(rand_score)
		#silhouette_score = metrics.silhouette_score(self.X,np.array(labels_pred), metric='euclidean')
		#print "Silhouette score is " + str(silhouette_score)
		[micro_f, macro_f] = self.F_Measure(labels_true,labels_pred)
		print "Micro F-Measure is " + str(micro_f)
		print "Macro F-Measure is " + str(macro_f)
		return micro_f, macro_f, mutual_info_score, rand_score

	def get_cluster_number_shift(self, labels_true, labels_pred):
		true_set = set(labels_true)
		pre_set = set(labels_pred)
		print pre_set
		dic = {} 
		for item in pre_set:
			dic[item] = {}
			for item_2 in true_set:
				dic[item][item_2] = 0

		for i in range(len(labels_true)):
			dic[labels_pred[i]][labels_true[i]] += 1
		print "ground truth data"
		print dic		
		final_dict = collections.defaultdict(dict)
		#used_list = set()
		for pred_key in pre_set:
			max_value = -1
			print dic[pred_key]
			for index, value in dic[pred_key].iteritems():
				#if index not in used_list:
				if value > max_value:
					max_label = index
					max_value = value
				final_dict[pred_key] = max_label
				#used_list.add(max_label)
		return final_dict


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
	parser.add_argument("datasets", choices=["zhihu","stackexchange","rottentomatoes","medhelp","test"], help="the dataset for experiments")
	parser.add_argument("clustering", choices=["wkmeans","kmeans","ahc"], help="the algorithm for clustering")
	parser.add_argument("features_type", choices=["tf-idf","binary"], help="the features type for clustering")
	parser.add_argument("test_type", choices=["train","cv"], help="clustering or cv?")
	#parser.add_argument('-w', action='store_true')
	# representation option for args
	args = parser.parse_args()
	features_type = args.features_type
	if args.datasets == "zhihu":
		num_clusters = 4
		#cluster_labels = pagesCluster(["../Crawler/crawl_data/Zhihu/"],num_clusters)
		cluster_labels = pagesCluster(["../Crawler/test_data/zhihu/"],num_clusters)
	elif args.datasets == "stackexchange":
		num_clusters = 5
		#cluster_labels = pagesCluster(["../Crawler/crawl_data/Questions/"],num_clusters)
		cluster_labels = pagesCluster(["../Crawler/test_data/stackexchange/"],num_clusters)
	elif args.datasets == "test":
		num_clusters = 5
		cluster_labels = pagesCluster(["../Crawler/test_data/train/"],num_clusters)
	elif args.datasets == "rottentomatoes":
		num_clusters = 7
		cluster_labels = pagesCluster(["../Crawler/test_data/rottentomatoes/"],num_clusters)
	elif args.datasets == "medhelp":
		num_clusters = 6
		cluster_labels = pagesCluster(["../Crawler/test_data/medhelp/"],num_clusters)
	else:
		print "error"

	
	if args.clustering == "kmeans":
		#cluster_labels.kmeans(cluster_labels.num_clusters,features_type,cv=True)
		cluster_labels.kmeans(cluster_labels.num_clusters,features_type)
		cluster_labels.Evaluation()
	elif args.clustering == "wkmeans":
		if args.test_type == "cv":
			cluster_labels.wkmeans(cluster_labels.num_clusters,features_type,cv=True,beta=2,replicates=10)
		else:
			cluster_labels.wkmeans(cluster_labels.num_clusters,features_type)
			cluster_labels.Evaluation()

	elif args.clustering == "ahc":
		cluster_labels.AgglomerativeClustering(cluster_labels.num_clusters)
		cluster_labels.Evaluation()

	#visualization
	v = visualizer(cluster_labels.UP_pages)
	twoD_file = "2Dfile_questions_Q7_norm_test.txt"
	v.show(v.UP_pages.ground_truth, v.UP_pages.category ,twoD_file)
	