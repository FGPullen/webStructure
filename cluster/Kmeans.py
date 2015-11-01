from page import Page
from pages import allPages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import cluster,metrics
from visualization import visualizer
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from time import time

class pagesCluster:
	def __init__(self, path_list):
		t0 = time()
		self.UP_pages = allPages(path_list)
		feature_matrix = []
		y =[]
		# get features and labels
		for page in self.UP_pages.pages:
			tfidf_vector = []
			for key in page.tfidf:
				#tfidf_vector.append(page.normonehot[key])
				tfidf_vector.append(page.normtfidf[key])
			feature_matrix.append(tfidf_vector)	

		y = self.UP_pages.category
		self.y = np.array(y)

		X = np.array(feature_matrix)
		X = scale(X)
		# select 
		#num_clusters = len(path_list)
		num_clusters = 7
		k_means = cluster.KMeans(n_clusters=num_clusters, n_init=10)
		k_means.fit(shuffle(X, random_state=0))
		zero_count = 0
		self.pre_y = k_means.predict(X)
		#print self.pre_y
		#print self.y
		print metrics.adjusted_mutual_info_score(self.y, self.pre_y)  
		print("done in %0.3fs." % (time() - t0))			

	def Output(self):
		write_file = open("cluster_result.txt","w")
		assert len(self.pre_y) == len(self.UP_pages.pages)
		for i in range(len(self.pre_y)):
			tmp = self.filename2Url(self.UP_pages.pages[i].path) + "\t" + str(self.pre_y[i])
			write_file.write(tmp + "\n")

	def filename2Url(self,filename):
		return filename.replace("_","/")


if __name__=='__main__':
	#cluster_labels = pagesCluster(["../Crawler/toy_data/users_toy/","../Crawler/toy_data/questions_toy/","../Crawler/toy_data/articles/","../Crawler/toy_data/lists/"])
	cluster_labels = pagesCluster(["../Crawler/crawl_data/Questions/"])
	print type(cluster_labels.pre_y)
	v = visualizer(cluster_labels.UP_pages)
	twoD_file = "2Dfile_questions_Q7_norm_test.txt"
	cluster_labels.Output()
	v.show(cluster_labels.pre_y,twoD_file)


