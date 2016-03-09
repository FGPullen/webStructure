from page import Page
from pages import allPages
import math
# This program aims to calculate the possibility of occurence of two xpaths. They are also regarded as non-consecutive bigram
# mutual infomation 
class bigram:

	def __init__(self,pages):
		self.pages = pages


	def find_bigram(self):

		df = self.pages.selected_df

		xpaths_dict = {}
		co_dict = {}
		bigram_list = []
		N_pages = len(self.pages.pages)

		for key1 in df.keys():
			xpaths_dict[key1] = {}
			co_dict[key1] = {}
			for key2 in df.keys():
				if key1 != key2:
					# if key occurs too little, we don't think it is a stop structure
					xpaths_dict[key1][key2] = float(N_pages**2)/(float(df[key1])*float(df[key2]))
					co_dict[key1][key2] = 0
					bigram_list.append([key1,key2])


		# unordered pair 
		for p in range(N_pages):
			page = self.pages.pages[p]
			for pair in bigram_list:
				key1, key2 = pair[0], pair[1]
				if page.xpaths[key1] >0 and page.xpaths[key2]>0:
					co_dict[key1][key2] += 1
					continue



		pair_dict = {}
		for key1 in xpaths_dict:
			for key2 in xpaths_dict[key1]:
				print xpaths_dict[key1][key2]
				p = float(co_dict[key1][key2])/float(N_pages)
				print p
				xpaths_dict[key1][key2] = p*xpaths_dict[key1][key2]
				
				if xpaths_dict[key1][key2] == 0:
					pair_dict["("+key1+","+key2+")"] = 0
				else:
					pair_dict["("+key1+","+key2+")"] = math.log(xpaths_dict[key1][key2]) * p
		bigram_list = []
		top = 1000
		pair_list = sorted(pair_dict.iteritems(),key=lambda x:x[1],reverse=True)
		for i in range(top):
			print pair_list[i][0] + "\t" + str(pair_list[i][1])
			[path1, path2] = pair_list[i][0].replace("(","").replace(")","").split(",")
			print str(df[path1]) + "\t" + str(df[path2])
			bigram_list.append([path1,path2])

		print bigram_list
		return bigram_list

if __name__ == "__main__":
	pages = allPages(["../Crawler/test_data/stackexchange/"])
	b = bigram(pages)
	b.find_bigram()