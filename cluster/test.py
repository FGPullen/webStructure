from page import Page
from cluster import cluster
from pages import allPages
import os
import math
import distance
from pages import allPages
from Kmeans import pagesCluster

if __name__=='__main__':
	num_clusters = 7
	clusters = []
	for i in range(1,num_clusters+1):
		clusters.append(cluster())
	cluster_labels = pagesCluster(["../Crawler/crawl_data/Questions/"],num_clusters)
	cluster_labels.DBSCAN()