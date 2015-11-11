from page import Page
from pages import allPages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn.cluster as Cluster
from visualization import visualizer
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from time import time
import operator
from cluster import cluster

class pagesCluster:
	def __init__(self, path_list,num_clusters):
		self.t0 = time()
		self.UP_pages = allPages(path_list)
		self.num_clusters = num_clusters
		self.clustering(num_clusters)
		
		
	def clustering(self,num_clusters):
		feature_matrix = []
		y =[]
		# get features and labels
		for page in self.UP_pages.pages:
			tfidf_vector = []
			for key in page.tfidf:
				#tfidf_vector.append(page.normonehot[key])
				tfidf_vector.append(page.normtfidf[key])
			feature_matrix.append(tfidf_vector)	

		self.X = np.array(feature_matrix)
		self.X = scale(self.X)
		# select 
		#num_clusters = len(path_list)
		k_means = Cluster.KMeans(n_clusters=num_clusters, n_init=10, random_state=0, n_jobs=2)
		k_means.fit(shuffle(self.X))
		self.pre_y = k_means.predict(self.X)
		self.UP_pages.updateCategory(self.pre_y)
		#print self.pre_y
		#print self.y
		#print metrics.adjusted_mutual_info_score(self.UP_pages.ground_truth,self.UP_pages.category)  
		print("done in %0.3fs." % (time() - self.t0))				

	def Output(self):
		write_file = open("cluster_result.txt","w")
		assert len(self.pre_y) == len(self.UP_pages.pages)
		for i in range(len(self.pre_y)):
			tmp = self.filename2Url(self.UP_pages.pages[i].path) + "\t" + str(self.pre_y[i])
			write_file.write(tmp + "\n")

	@staticmethod
	def Precision_Recall_F(labels_true,labels_pred):
		ground_truth_set = set(labels_true)
		labels_set = set(labels_pred)
		# dict with index and cluster_index:
		labels = {}
		for item in ground_truth_set:
			labels[item] = {}
			for item2 in labels_set:
				labels[item][item2] = 0
		# get the distribution of clustering results
		for i in range(len(labels_true)):
			g_index = labels_true[i]
			c_index = labels_pred[i]
			labels[g_index][c_index] += 1
		# get the statistical results
		for item in ground_truth_set:
			total_number = 0
			for num in labels[item]:
				total_number+=labels[item][num]
			sorted_list = sorted(labels[item].iteritems(), key=lambda d:d[1], reverse = True)
			print "For ground truth is " + str(item) + " which has " + str(total_number) + " the biggest clsuter has " + str(sorted_list[0][1]) 



		return [precision, recall, f1]


	def Evaluation(self):
		labels_true = self.UP_pages.ground_truth
		labels_pred = self.UP_pages.category
		pages = self.UP_pages
		#self.Precision_Recall_F(labels_true,labels_pred)
		print "Mutual Info Score is " + str(metrics.adjusted_mutual_info_score(labels_true, labels_pred))
		print "Adjusted Rand Score is " + str(metrics.adjusted_rand_score(labels_true, labels_pred))
		silhouette_score = metrics.silhouette_score(self.X,np.array(labels_pred), metric='euclidean')
		print "Silhouette score is " + str(silhouette_score)


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
	#cluster_labels = pagesCluster(["../Crawler/toy_data/users_toy/","../Crawler/toy_data/questions_toy/","../Crawler/toy_data/articles/","../Crawler/toy_data/lists/"])
	num_clusters = 5
	clusters = []
	for i in range(1,num_clusters+1):
		clusters.append(cluster())
	cluster_labels = pagesCluster(["../Crawler/crawl_data/Questions/"],num_clusters)
	cluster_labels.Evaluation()
	cluster_labels.get_top_local_xpath(0.9, "user")

	'''
	visualization
	'''
	#v = visualizer(cluster_labels.UP_pages)
	#twoD_file = "2Dfile_questions_Q7_norm_test.txt"
	#cluster_labels.Output()
	#v.show(v.UP_pages.ground_truth,twoD_file)


