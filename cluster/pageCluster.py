from pages import allPages
import sys
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
from cluster import cluster

class pagesCluster:
	def __init__(self, path_list,num_clusters):
		self.t0 = time()
		self.UP_pages = allPages(path_list)
		self.num_clusters = num_clusters
		#self.clustering(num_clusters)
		
		
	def kmeans(self,num_clusters):
		feature_matrix = []
		y =[]
		# get features and labels
		
		for page in self.UP_pages.pages:
			#use tf-idf sth
			'''
			tfidf_vector = []
			for key in page.tfidf:
				#tfidf_vector.append(page.normonehot[key])
				tfidf_vector.append(page.tfidf[key])
			tfidf_vector = normalize(tfidf_vector,norm='l1')[0]
			feature_matrix.append(tfidf_vector)	
			'''
			#feature_matrix.append(page.embedding)

			# selected normalized tf idf 
			
			vector = []
			for key in page.selected_tfidf:
				vector.append(page.selected_tfidf[key])
			vector = normalize(vector,norm='l1')[0]
			feature_matrix.append(vector)
			
			# Leung Baseline
			'''
			vector = []
			for key in page.Leung:
				#print key + "\t" + str(page.Leung[key])
				vector.append(page.Leung[key])
			vector = normalize(vector,norm='l1')[0]
			feature_matrix.append(vector)
			'''

		self.X = np.array(feature_matrix)
		#self.X = scale(self.X)
		# select 
		#num_clusters = len(path_list)
		k_means = Cluster.KMeans(n_clusters=num_clusters, n_init=10, random_state=0, n_jobs=2)
		#k_means.fit(shuffle(self.X))
		#self.pre_y = k_means.predict(self.X)
		k_means.fit(self.X)
		self.pre_y = k_means.labels_
		self.UP_pages.updateCategory(self.pre_y)
		#print self.pre_y
		#print self.y
		#print metrics.adjusted_mutual_info_score(self.UP_pages.ground_truth,self.UP_pages.category)  
		#print("done in %0.3fs." % (time() - self.t0))				

	def AgglomerativeClustering(self, num_clusters):
		'''
		feature_matrix = []
		y =[]
		for page in self.UP_pages.pages:
			vector = []
			for key in page.Leung:
				#print key + "\t" + str(page.Leung[key])
				vector.append(page.Leung[key])
			vector = normalize(vector,norm='l1')[0]
			feature_matrix.append(vector)

		self.X = np.array(feature_matrix)
		ahc = Cluster.AgglomerativeClustering(n_clusters=num_clusters,linkage='complete')
		ahc.fit(self.X)
		self.pre_y = ahc.labels_
		self.UP_pages.updateCategory(self.pre_y)
		'''
		#self.X = np.array([[0,1,2,4],[1,0,3,4],[2,3,0,1],[4,4,1,0]])
		self.X = self.get_affinity_matrix()
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
				ng[g_index] = 0
			if nc.has_key(c_index):
				nc[c_index] += 1
			else:
				nc[c_index] = 0
		# get the statistical results
		for i in ground_truth_set:
			for j in labels_set:
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


	def Evaluation(self):
		labels_true = self.UP_pages.ground_truth
		labels_pred = self.UP_pages.category
		print "We have %d pages for ground truth!" %(len(labels_true))
		print "We have %d pages after prediction!" %(len(labels_pred))
		assert len(labels_true) == len(labels_pred)
		pages = self.UP_pages
		#self.Precision_Recall_F(labels_true,labels_pred)
		print "Mutual Info Score is " + str(metrics.adjusted_mutual_info_score(labels_true, labels_pred))
		print "Adjusted Rand Score is " + str(metrics.adjusted_rand_score(labels_true, labels_pred))
		silhouette_score = metrics.silhouette_score(self.X,np.array(labels_pred), metric='euclidean')
		print "Silhouette score is " + str(silhouette_score)
		[micro_f, macro_f] = self.F_Measure(labels_true,labels_pred)
		print "Micro F-Measure is " + str(micro_f)
		print "Macro F-Measure is " + str(macro_f)


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
	parser.add_argument("datasets", choices=["zhihu","stackexchange","test"], help="the dataset for experiments")
	parser.add_argument("clustering", choices=["kmeans","ahc"], help="the algorithm for clustering")
	# representation option for args
	args = parser.parse_args()
	if args.datasets == "zhihu":
		num_clusters = 6
		cluster_labels = pagesCluster(["../Crawler/crawl_data/Zhihu/"],num_clusters)
	elif args.datasets == "stackexchange":
		num_clusters = 7
		cluster_labels = pagesCluster(["../Crawler/crawl_data/Questions/"],num_clusters)
	elif args.datasets == "test":
		num_clusters = 7
		cluster_labels = pagesCluster(["../Crawler/crawl_data/test/"],num_clusters)
	else:
		print "error"

	
	if args.clustering == "kmeans":
		cluster_labels.kmeans(cluster_labels.num_clusters)
		cluster_labels.Evaluation()
	elif args.clustering == "ahc":
		cluster_labels.AgglomerativeClustering(cluster_labels.num_clusters)
		cluster_labels.Evaluation()

	#visualization
	v = visualizer(cluster_labels.UP_pages)
	twoD_file = "2Dfile_questions_Q7_norm_test.txt"
	v.show(v.UP_pages.ground_truth, v.UP_pages.category ,twoD_file)
	