from page import Page
from pages import allPages
# This program aims to calculate the possibility of occurence of two xpaths. They are also regarded as non-consecutive bigram

if __name__=='__main__':
	UP_pages = allPages(["../Crawler/crawl_data/Questions/"])
	idf = UP_pages.nidf


	xpaths_dict = {}
	co_dict = {}
	for key1 in idf.keys():
		xpaths_dict[key1] = {}
		co_dict[key1] = {}
		for key2 in idf.keys():
			if key1 != key2:
				xpaths_dict[key1][key2] = 1.0/(float(idf[key1])*float(idf[key2]))
				co_dict[key1][key2] = 0


	N_pages = len(UP_pages.pages)
	for p in range(N_pages):
		page = UP_pages.pages[p]
		occur_list = []
		for item in page.xpaths:
			if page.xpaths[item]>0:
				occur_list.append(item)
		for i in range(len(occur_list)):
			for j in range(i+1,len(occur_list)):
				key1 = occur_list[i]
				key2 = occur_list[j]
				#print xpaths_dict[key1][key2]
				co_dict[key1][key2] += 1


	for key1 in xpaths_dict:
		for key2 in xpaths_dict[key1]:
			xpaths_dict[key1][key2] *= co_dict[key1][key2]
			if xpaths_dict[key1][key2]!=0:
				print str(xpaths_dict[key1][key2]) + "\t" +key1+ "\t" +key2 

