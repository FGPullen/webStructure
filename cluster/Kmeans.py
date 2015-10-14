from page import Page
from pages import allPages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.preprocessing import scale

if __name__=='__main__':
	UP_pages = allPages(["../Crawler/toy_data/users/","../Crawler/toy_data/questions/","../Crawler/toy_data/articles/"])
	feature_matrix = []
	for page in UP_pages.pages:
		tfidf_vector = []
		for key in page.tfidf:
			tfidf_vector.append(page.tfidf[key])
		feature_matrix.append(tfidf_vector)
	X = np.array(feature_matrix)
	X = scale(X)
	print type(X)
	k_means = cluster.KMeans(n_clusters=3, n_init=10)
	k_means.fit(X)
	zero_count = 0
	labels = k_means.labels_
	for label in labels:
		print label
	#for label in labels:
	#	print label