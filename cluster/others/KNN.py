from page import Page
from pages import allPages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import cluster,metrics,cross_validation,neighbors
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from time import time

class pagesCluster:
	def __init__(self, path_list):
		t0 = time()
		UP_pages = allPages(path_list)
		feature_matrix = []
		y =[]
		# get features and labels
		for page in UP_pages.pages:
			tfidf_vector = []
			for key in page.tfidf:
				tfidf_vector.append(page.tfidf[key])
			feature_matrix.append(tfidf_vector)	

		y = UP_pages.category
		self.y = np.array(y)
		X = np.array(feature_matrix)
		X = scale(X)

		X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.99, random_state=2)
		
		clf = neighbors.KNeighborsClassifier(1)
		clf.fit(X,y)
		labels = clf.predict(X_test)
		right = 0
		for i in range(len(labels)):
			if labels[i] == y_test[i]:
				right += 1
		print "accuracy is "+ str(float(right)/float(len(y_test)))
		# select  
		print("done in %0.3fs." % (time() - t0))		

if __name__=='__main__':
	cluster_labels = pagesCluster(["../Crawler/toy_data/users/","../Crawler/toy_data/questions/"])


