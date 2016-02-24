#import os.path, sys
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from HITS import HITS
from pages import allPages
import numpy as np
from sklearn.preprocessing import normalize

class sitemap:
	def __init__(self):
		self.get_trans_mat()
		self.write()

	def get_trans_mat(self):
		self.pages = allPages(["../Crawler/test_data/stackexchange/"])
		cluster_num = 4
		self.cluster_num = cluster_num
		trans_mat = np.zeros((cluster_num,cluster_num))
		total_links = 0
		for page in self.pages.pages:
			group = self.annotate(page.path,False)
			if group == 0:
				continue
			print page.path , group

			link_dict = page.getAnchor()
			for key,links in link_dict.iteritems():
				for link in links:
					tag = self.annotate(link,True)
					if tag!=0:
						total_links += 1
						trans_mat[group-1,tag-1] += 1
						if group == tag:
							print link , page.path
		print trans_mat
		trans_mat = normalize(trans_mat,norm='l1', axis=1)
		print trans_mat
		print "total_links has " + str(total_links)
		self.trans_mat = trans_mat



	def annotate(self,path,anchor):
		path = path.replace("../Crawler/test_data/stackexchange/", "")
		if "http" in path and anchor: 
			tag = 0
		elif "/users/" in path:
			tag = 1
		elif "/questions/tagged/" in path:
			tag = 3
		elif "/questions/" in path or "/q/" in path or "/a/" in path:
			tag = 2
		#elif "/tags/" in path:
		#	tag = 6
		elif "/posts/" in path:
			tag = 4
		elif "/feeds/" in path:
			tag = 0
		else:
			return 0
		return tag
	
	'''		
	def annotate(self,path):
		path = path.replace("../Crawler/test_data/rottentomatoes/","")
		if "/top/" in path:
			tag = 3
		elif "/celebrity/" in path:
			if "/pictures/" in path:
				tag = 6
			else:
				tag = 1
		elif "/critic/" in path:
				tag = 2
		elif "/m/" in path or "/tv/" in path:
			if "/trailers/" in path:
				tag = 5
			elif "/pictures/" in path:
				tag = 6
			else:
				tag = 4
		else: # guide
			tag =0
		return tag
	
	def annotate(self,path):
		path = path.replace("../Crawler/test_data/zhihu/","")
		if "follow" in path:
			tag = 3
		elif "/people/" in path:
			tag = 1
		elif "/question/" in path:
				tag = 2
		elif "/topic/" in path:
			tag = 4
		elif "/collection/" in path:
			tag = 5
		else:
			tag = 0
		return tag
	'''
	#def compute_hits(self):
	def write(self):
		file = open("./Hits/stackexchange_mat.txt","w")
		for i in range(self.cluster_num):
			for j in range(self.cluster_num):
				file.write(str(i+1) + " " + str(j+1) + " " + str(self.trans_mat[i,j]) + "\n")



if __name__ == "__main__":
	print "hello"
	s = sitemap()