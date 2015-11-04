from page import Page
from pages import allPages
from Kmeans import pagesCluster
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import cluster,metrics
from visualization import visualizer
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from time import time

if __name__=='__main__':
	#cluster_labels = pagesCluster(["../Crawler/toy_data/users_toy/","../Crawler/toy_data/questions_toy/","../Crawler/toy_data/articles/","../Crawler/toy_data/lists/"])
	cluster_labels = pagesCluster(["../Crawler/crawl_data/Questions/"])
	pages = cluster_labels.UP_pages
	for i in range(len(pages.category)):
		print pages.pages[i].path + "\t" + str(pages.category[i])
