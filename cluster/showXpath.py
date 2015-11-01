from page import Page
from pages import allPages
# this program aims to find out the xpath frequency differences between two page objects.

if __name__=='__main__':
	UP_pages = allPages(["../Crawler/crawl_data/showXpath/"])

	a = UP_pages.pages[0] # less
	b = UP_pages.pages[1] # more
	print a.path + "\t" + b.path
	dif_dict = {}
	for item in a.xpaths:
		dif_dict[item] = a.xpaths[item]-b.xpaths[item]

	# Top 10 i > j
	g_sorted_result_dict= sorted(dif_dict.iteritems(), key=lambda d:d[1], reverse = True)
	l_sorted_result_dict= sorted(dif_dict.iteritems(), key=lambda d:d[1], reverse = False)
	for i in range(10):
		print str(g_sorted_result_dict[i][0]) + "\t" + str(g_sorted_result_dict[i][1])
	for i in range(10):
		print str(l_sorted_result_dict[i][0]) + "\t" + str(l_sorted_result_dict[i][1])