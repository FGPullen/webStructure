from page import Page
import os
import math

class allPages:
	def __init__(self,path_list):
		self.pages = []
		self.category = []
		self.full_xpaths = []
		self.idf = {}
		self.nidf = {}
		self.addPages(path_list)
		self.expandXpaths()
		self.updateidf()
		self.updatetfidf()
	
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
		'''
		x1 = "/html/body/div/div/div/div/div/div/div/h3/span"
		x2 = "/html/body/div/div/div/div/div/div/div/h3"
		x3 = "/html/body/div/div/div/div/div/div/div/div/div/a/div/img"
		for page in self.pages:
			page.xpaths[x1] *= 100
			page.xpaths[x2] *= 100
			page.xpaths[x3] *= 100
		'''

	def updatetfidf(self):
		for page in self.pages:
			page.updatetfidf(self.idf)

	#def getFarpair(self):


def distance(page1,page2):
	dis = 0.0
	for item in page1.tfidf:
		dis += math.pow((page1.normtfidf[item]-page2.normtfidf[item]),2)
	return math.sqrt(dis)

if __name__=='__main__':
	UP_pages = allPages(["../Crawler/toy_data/users/","../Crawler/toy_data/questions/","../Crawler/toy_data/lists/"])
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
	