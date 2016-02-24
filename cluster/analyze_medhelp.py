import numpy as np
from page import Page
from cluster import cluster
import numpy as np
from pages import allPages
import os
from sklearn.preprocessing import scale,normalize
import math
import distance
import collections


if __name__ == "__main__":
	two_dimension_lines = open("./results/2D_plot_file.txt","r").readlines()
	cluster_dict = {}
	for line in two_dimension_lines:
		[doc_name, cluster, x, y] = line.strip().split("\t")
		cluster_dict[doc_name] = cluster
	print cluster_dict

	# here we only have points for gorup/forum 
	truth_feature_dict = {}
	truth_feature_lines = open("./results/medhelp-binary.csv","r").readlines()
	for line in truth_feature_lines:
		tmp = line.strip().split("\t")
		doc_name = tmp[0]
		features = tmp[1:]
		for i in range(len(features)):
			features[i] = float(features[i])
		truth_feature_dict[doc_name] = features


	m3 = []
	m2 = []

	for key in truth_feature_dict:
		if "/forums/" in key or "/group/" in key:  # 2 & 3
			if cluster_dict[key] == "2":
				m2.append(truth_feature_dict[key])
			elif cluster_dict[key] == "3":
				m3.append(truth_feature_dict[key])

	m2_ = np.array(m2)
	m3_ = np.array(m3)
	length = m2_.shape[1]
	print m2_.shape[1]
	print m3_.shape[1]
	assert m2_.shape[1] == m3_.shape[1]
	avg_m2 = m2_.sum(axis=0)/float((m2_.shape[0]))
	avg_m3 = m3_.sum(axis=0)/float((m3_.shape[0]))

	print avg_m2
	print avg_m3
    # 0 - 237
	pages = allPages(["../Crawler/test_data/medhelp/"])
	#pages.find_important_xpath()
	page = pages.pages[0]
	keys = page.selected_tfidf.keys()

	for i in range(length):
		if avg_m2[i] == 0 and (avg_m3[i]!=0):
			print str(keys[i]) + " not in 2 " + str(avg_m3[i])
		if avg_m3[i] == 0 and (avg_m2[i]!=0):
			print str(keys[i]) + " not in 3 " + str(avg_m2[i])
 
 	print " ================= for date spane ==================" 
 	span = "div/div/div/div/div/span"
 	for index, key in enumerate(keys):
 		if key.endswith(span):
 			print str(avg_m2[index]) + " " + str(avg_m3[index]) + " " + str(key)

