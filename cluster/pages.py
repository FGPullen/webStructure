from page import Page
from cluster import cluster
import os
import math
import distance

class allPages:
	def __init__(self,path_list,dm="no"):
		self.pages = []
		self.category = []
		self.full_xpaths = []
		self.ground_truth = []
		self.idf = {}
		self.nidf = {}
		self.addPages(path_list)
		self.expandXpaths()
		self.updateidf()
		self.get_ground_truth()
		#self.top_local_stop_structure_gt(0.9)
		self.heuristic_weight()
		self.updatetfidf()
		if dm=="yes":
			self.getDistanceMatrix("./Data/edit_distance.matrix")
		
	
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
			self.nidf[xpath] = 0
		# count document frequency
		for page in self.pages:
			for xpath in self.full_xpaths:
				if page.xpaths[xpath] !=0:
					self.nidf[xpath] +=1
		# log(n/N)
		for xpath in self.full_xpaths:
			self.idf[xpath] = math.log((float(N))/(float(self.nidf[xpath])),2)
		
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
		print "number of cluster 5 is " + str(len(gt_clusters[6].pages))
		gt_clusters[6].find_local_stop_structure(self.nidf,global_threshold)



		
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

	def updatetfidf(self):
		for page in self.pages:
			page.updatetfidf(self.idf)

	# update category based on predicted y
	def updateCategory(self,pred_y):
		assert len(self.category) == len(pred_y)
		for i in range(len(pred_y)):
			self.category[i] = pred_y[i]

	def getDistanceMatrix(self,write_path):
		self.dist_matrix = []
		write_file = open(write_path,"w")
		for i in range(len(self.pages)):
			print "calculate " + str(i)
			s_i = self.pages[i].dfs_xpaths_list
			tmp_list = []
			for j in range(len(self.pages)):
				if i==j:
					tmp_dis = 0
				elif i<j:
					s_j = self.pages[j].dfs_xpaths_list
					tmp_dis = int(distance.levenshtein(s_i,s_j))
				else:
					tmp_dis = self.dist_matrix[j][i]
				tmp_list.append(tmp_dis)
				print "calculate " + str(j) + "\t" + str(tmp_dis)
			self.dist_matrix.append(tmp_list)
			for item in tmp_list:
				write_file.write(str(item) + "\t")
			write_file.write("\n")


	def get_ground_truth(self):
		# /users/ /questions/ /q/ /questions/tagged/   /tags/ /posts/ /feeds/ /others
		for i in range(len(self.pages)):
			path = self.pages[i].path.replace("_","/")
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

def page_distance(page1,page2):
	dis = 0.0
	for item in page1.tfidf:
		dis += math.pow((page1.normtfidf[item]-page2.normtfidf[item]),2)
	return math.sqrt(dis)

if __name__=='__main__':
	#UP_pages = allPages(["../Crawler/crawl_data/Questions/"])
	pages = allPages(["../Crawler/crawl_data/Questions/"]).pages
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
	