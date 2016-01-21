from page import Page
from cluster import cluster
import numpy as np
import os
from sklearn.preprocessing import scale,normalize
import math
import distance

class allPages:
	def __init__(self, folder_path ,dm="no"):
		self.folder_path = folder_path
		print folder_path
		# initialize data structure
		self.pages = []
		self.category = []
		self.full_xpaths = []
		self.ground_truth = []
		self.idf = {}
		self.selected_idf = {}
		self.df = {}

		#  update attributes
		self.addPages(folder_path)
		self.expandXpaths()
		self.updateidf()
		self.get_ground_truth()
		self.num = len(self.pages)
		#self.top_local_stop_structure_gt(0.9)
		#self.heuristic_weight_zhihu()
		#self.heuristic_weight()
		self.updatetfidf()
		self.filter_idf(0.01,1.0)
		self.filter_dfs_xpaths_list()
		self.Leung_baseline()
		self.selected_tfidf()
		if dm=="yes":
			self.getDistanceMatrix("./Data/edit_distance.matrix")
		
	def filter_dfs_xpaths_list(self):
		print "start filtering for xpaths list"
		for page in self.pages:
			page.filtered_dfs_xpaths_list = [item for item in page.dfs_xpaths_list if item in self.selected_idf]
			#print "we filterd " + str(len(page.dfs_xpaths_list) - len(page.filtered_dfs_xpaths_list))
		print "end filtering for xpaths list"

	def update_full_xpaths(self,_page_):
		for xpath in _page_.xpaths.keys():
			if xpath not in self.full_xpaths:
				self.full_xpaths.append(xpath)

	def addPages(self,folder_path_list):
		category_num = 0
		for folder_path in folder_path_list:
			folder_pages = []
			for html_file in os.listdir(folder_path):
				if ".DS_Store" not in html_file:
					file_path = folder_path + html_file
					file_page = Page(file_path)
					# the same number for pags & category
					self.pages.append(file_page)
					self.category.append(category_num)
					self.update_full_xpaths(file_page)
			category_num+=1

	def expandXpaths(self):
		for page in self.pages:
			page.expandXpaths(self.full_xpaths)

	def updateidf(self):
		N = len(self.pages)
		# initiate
		for xpath in self.full_xpaths:
			self.df[xpath] = 0
		# count document frequency
		for page in self.pages:
			for xpath in self.full_xpaths:
				if page.xpaths[xpath] !=0:
					self.df[xpath] +=1
		# log(n/N)
		for xpath in self.full_xpaths:
			self.idf[xpath] = math.log((float(N))/(float(self.df[xpath])),2)
			# add sqrt into idf so that calculating distance there will be only one power of idf
			#self.idf[xpath] = math.sqrt(self.idf[xpath])
		
	def top_local_stop_structure_gt(self,threshold):
		global_threshold = len(self.pages) * threshold
		gt_clusters = []
		for item in set(self.ground_truth):
			gt_clusters.append(cluster())
			for i in range(len(self.ground_truth)):
				if self.ground_truth[i] == item:
					gt_clusters[item].addPage(self.pages[i])
			print str(item) + "\t" + str(len(gt_clusters[item].pages)) 

		print "number of gt cluster is " + str(len(gt_clusters))
		print "number of cluster 5 is " + str(len(gt_clusters[4].pages))
		gt_clusters[4].find_local_stop_structure(self.df,global_threshold)

	def affinity(self,page1,page2):
		distance = 0.0
		s,t,m =0.0,0.0,0.0
		for key in self.selected_idf:
			if page1.xpaths[key] >= 1:
				s += 1
			if page2.xpaths[key] >= 1:
				t += 1
			if page1.xpaths[key] >= 1 and page2.xpaths[key] >= 1:
				m += 1
		if s+t-m == 0:
			distance = 0.0
		else:
			distance = float(m)/float(s+t-m)
		return distance

		
	def heuristic_weight(self):
		# user 
		x = []
		x.append("/html/body/div/div/div/div/div/div/div/div/div/div/ul/li/span")
		x.append("/html/body/div/div/div/div/div/a/div")
		x.append("/html/body/div/div/noscript/div/img")
		x.append("/html/body/div/div/div/div/div/div/div/h3/span")
		x.append("/html/body/div/div/div/div/div/div/div/ul/li/a/span")

		# Question
		x.append("/html/body/div/div/div/link")
		x.append("/html/body/div/div/div/div/div/ul/li/div")
		x.append("/html/body/div/div/div/div/div/table/tr/td/div/div/b")
		x.append("/html/body/div/div/div/div/div/table/tr/td/div/table/tr/td/div/div/br")
		x.append("/html/body/div/div/div/div/div/table/tr/td/div/input")

		#list 
		x.append("/html/body/div/div/div/div/div/div/div")
		x.append("/html/body/div/div/div/div/div/div/h3/a")
		x.append("/html/body/div/div/div/div/ul/li/div")
		x.append("/html/body/div/div/div/div/h4/a")
		x.append("/html/body/div/div/div/div/div/div/div/div/div/br")

		# tag 
		x.append("/html/body/div/div/div/div/br")
		x.append("/html/body/div/div/div/div/h2/a")

		# post 
		x.append("/html/body/div/div/div/div/div/p/i")
		x.append("/html/body/div/div/div/div/div/p/a")
		x.append("/html/body/div/div/div/form/div/div/span")

		# feeds
		x.append("/html/body/feed/subtitle")
		x.append("/html/body/feed/entry/link")
		x.append("/html/body/feed/entry/category")

		for page in self.pages:
			for item in x:
				page.xpaths[item] *= 20

	def heuristic_weight_zhihu(self):
		# 0 for people
		x = []
		x.append("/html/body/div/div/div/div/div/a/span")
		x.append("/html/body/div/div/div/div/div/div/div/div/div/div/span/span")
		x.append("/html/body/div/div/div/div/div/div/div/div/div/div/span/input")
		x.append("/html/body/div/div/div/div/div/div/span/span")
		x.append("/html/body/div/div/div/a/br")

		# Question
		x.append("/html/body/div/div/div/div")
		x.append("/html/body/div/div/div/div/button")
		x.append("/html/body/div/div/div/div/div")

		#list 
		x.append("/html/body/div/div/div/div/div/form/input")
		x.append("/html/body/div/div/div/div/div/form/div/div/div/img")
		#x.append("/html/body/div/div/div/div/div/form/div/a")

		# tag 
		x.append("/html/body/div/div/div/div/div/div/div/div/div/div/div/div/span/span")
		x.append("/html/body/div/div/div/div/div/div/div/div/div/div/div/div/span/span/a")
		#x.append("/html/body/div/div/div/div/div/div/div/div/div/div/div/div/a/i")
		# post 
		x.append("/html/body/div/div/div/div/div/h2/a")
		x.append("/html/body/div/div/div/div/div/div/div/div/textarea/span/a")

		for page in self.pages:
			for item in x:
				page.xpaths[item] *= 20

	def get_affinity_matrix(self):
		# calculate affinity matrix based on Xpath
		# if we assign a matrix before , would it be much quicker?
		print len(self.selected_idf)
		matrix = np.zeros(shape=(self.num,self.num))
		for i in range(self.num):
			for j in range(i+1,self.num):
				matrix[i,j] = self.affinity(self.pages[i],self.pages[j])
				matrix[j,i] = matrix[i,j]
		return matrix

	def get_one_hot_distance_matrix(self):
		matrix = np.zeros(shape=(self.num,self.num))
		for i in range(self.num):
			temp = []
			for key, value in self.pages[i].Leung.iteritems():
				temp.append(value)
			s_i  = np.array(normalize(temp,norm='l1')[0])
			for j in range(i+1,self.num):
				temp = []
				for key, value in self.pages[j].Leung.iteritems():
					temp.append(value)
				s_j  = np.array(normalize(temp,norm='l1')[0])
				matrix[i,j] =  np.linalg.norm(s_i - s_j)
				matrix[j,i] = matrix[i,j]
		return matrix



	def updatetfidf(self):
		for page in self.pages:
			page.updatetfidf(self.idf)

	# update category based on predicted y
	def updateCategory(self,pred_y):
		assert len(self.category) == len(pred_y)
		for i in range(len(pred_y)):
			self.category[i] = pred_y[i]

	def get_edit_distance_matrix(self):
		matrix = np.zeros(shape=(self.num,self.num))
		for i in range(self.num):
			if i % 8 == 0:
				print "finish " + str(float(i)/float(self.num)) + "% of " + str(self.num) + " pages"
			s_i = self.pages[i].filtered_dfs_xpaths_list
			for j in range(i+1,self.num):
					s_j = self.pages[j].filtered_dfs_xpaths_list
					matrix[i][j] = int(distance.levenshtein(s_i,s_j))
					matrix[j][i] = matrix[i][j]
					print str(i) + "\t" + str(j)
					#print str(len(s_i)) + "\t" + str(len(s_j))
		return matrix

	def read_edit_distance_matrix(self):
		matrix = np.zeros(shape=(self.num,self.num))
		lines = open("./Data/edit_distance.matrix","r").readlines()
		for i in range(self.num):
			line = lines[i].strip().split("\t")
			assert len(line) == self.num
			for j,item in enumerate(line):
				matrix[i][j] = float(item)
		return matrix

	def get_ground_truth(self):
		# /users/ /questions/ /q/ /questions/tagged/   /tags/ /posts/ /feeds/ /others
		if "../Crawler/crawl_data/Questions/"  in self.folder_path or "../Crawler/crawl_data/test/" in self.folder_path or "../Crawler/test_data/stackexchange/" in self.folder_path:
			print "????"
			for i in range(len(self.pages)):
				path = self.pages[i].path.replace("../Crawler/crawl_data/Questions/", "")
				if "/users/" in path:
					tag = 1
				elif "/questions/tagged/" in path:
					tag = 3
				elif "/questions/" in path or "/q/" in path or "/a/" in path:
					tag = 2
				elif "/tags/" in path:
					tag = 4
				elif "/posts/" in path:
					tag = 5
				elif "/feeds/" in path:
					tag = 6
				else:
					tag = 0
				#print "tag is " + str(tag)
				self.ground_truth.append(tag)
		# zhihu
		# /people/  /question/ /question/answer/ /topic/  (people/followed/ people/follower/ -> index ) /ask /collection
		elif "../Crawler/crawl_data/Zhihu/" in self.folder_path or "../Crawler/test_data/zhihu/" in self.folder_path:
			print "!!!!"
			for i in range(len(self.pages)):
				path = self.pages[i].path.replace("../Crawler/test_data/zhihu/","")
				if "follow" in path:
					tag = 2
				elif "/people/" in path:
					tag = 0
				elif "/question/" in path:
						tag = 1
				elif "/topic/" in path:
					tag = 3
				elif "/collection/" in path:
					tag = 4
				else:
					tag =5
				self.ground_truth.append(tag)
		

	def Leung_baseline(self):
		# one-hot representation
		# threshold set to be 0.25 which means that xpath appear over 25% pages will be kept.
		N = self.num
		for key in self.idf:
			#print key
			if float(self.df[key])/float(N) >= 0.01:
				for page in self.pages:
					page.update_Leung(key)

	def filter_idf(self,min_t, max_t):
		# aim to get a subset of xpaths which filters out path that appears in very few documents
		# as well as those that occur in almost all documents
		for key in self.idf:
			if float(self.df[key])/float(self.num) >= min_t:
				if float(self.df[key])/float(self.num) <= max_t:
					self.selected_idf[key] = self.df[key]

	def selected_tfidf(self):
		N = self.num
		for key in self.idf:
			if float(self.df[key])/float(N) >= 0.01:
				for page in self.pages:
					page.update_selected_tfidf(key)
				




if __name__=='__main__':
	#UP_pages = allPages(["../Crawler/crawl_data/Questions/"])
	#pages = allPages(["../Crawler/crawl_data/Questions/"])
	pages = allPages(["../Crawler/test_data/stackexchange/"])
	print "numer of pages " + str(len(pages.pages))
	print "number of xpath " + str(len(pages.idf))
	xpath_list = ['/html/body/div/div/div/div/div/div/div/div/div/a/div/img']
	for xpath in xpath_list:
		print "---- " + xpath + " -----"
		print pages.df[xpath]
		for page in pages.pages:
			if xpath in page.xpaths_list:
				print page.path
		print "------------------------"


	'''
	depth = []
	for key in pages.idf:
		depth.append(key.count("/"))

	for page in pages.pages:
		for xpath in page.xpaths:
			if xpath.count("/") >110 and  page.tfidf[xpath]>0:
				print page.path
				print xpath
	average = sum(depth) / len(depth)
	max_depth = max(depth)
	print "average depth of xpath is " + str(average)
	print "max depth of xpath is " + str(max_depth)
	g_dict = {}
	for key in pages.ground_truth:
		if key not in g_dict:
			g_dict[key] = 1
		else:
			g_dict[key] += 1
	for key in g_dict:
		g_dict[key] = float(g_dict[key])/float(len(pages.pages))
	print g_dict
	
	length = len(pages.df)
	threshold = [0.0,0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]
	for t in threshold:
		count = 0
		for xpath in pages.df:
			if float(pages.df[xpath])/float(length) >= t:
				count += 1
		print float(count)/float(length)


	
	edit_lines = open("./Data/edit_distance.matrix").readlines()
	length = len(edit_lines) - 2
	for i in range(length):
		print "=================="
		print pages[i].path
		dis = edit_lines[i].strip().split("\t")
		sub_dis = {}
		for j in range(length):
			sub_dis[j] = dis[j]
			sorted_dict= sorted(sub_dis.iteritems(), key=lambda d:d[1], reverse = False)
		
		# find the top 10 , the first one will be it self
		
		for j in range(1,11):
			print str(sorted_dict[j][1]) + "\t" + pages[sorted_dict[j][0]].path
		print "================="
	'''





	'''
	page1 = UP_pages.pages[451]
	page2 = UP_pages.pages[452]
	x1 = page1.dfs_xpaths_list
	x2 = page2.dfs_xpaths_list
	print page1.path
	print page2.path
	d = distance.levenshtein(x1,x2)
	print d
	'''
	# portiaon of d/ min(len(x1),len(x2))

	'''
	count_zero = 0
	count_one = 0
	for item in UP_pages.category:
		if item == 0:
			count_zero+=1
		elif item == 1:
			count_one+=1
	max_D = 0
	min_D = 999
	page_pair = []
	for i in range(count_zero):
		for j in range(i,count_zero):
			D = distance(UP_pages.pages[i],UP_pages.pages[j])
			if D >max_D:
				max_D = D 
				print max_D
				#print str(i) + "\t" + str(j)
				page_pair = [i,j]


	i = page_pair[0]
	j = page_pair[1]
	print str(i) + "\t" + str(j)
	print UP_pages.pages[i].path, UP_pages.pages[j].path
	dif_dict = {}
	for item in UP_pages.pages[i].tfidf:
		dif_dict[item] = UP_pages.pages[i].normtfidf[item]-UP_pages.pages[j].normtfidf[item]

	# Top 10 i > j
	g_sorted_result_dict= sorted(dif_dict.iteritems(), key=lambda d:d[1], reverse = True)
	l_sorted_result_dict= sorted(dif_dict.iteritems(), key=lambda d:d[1], reverse = False)
	for i in range(10):
		print str(g_sorted_result_dict[i][0]) + "\t" + str(g_sorted_result_dict[i][1])
		print str(l_sorted_result_dict[i][0]) + "\t" + str(l_sorted_result_dict[i][1])
	# Top 10 j > i
	'''
	