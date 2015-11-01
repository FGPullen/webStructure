from page import Page
from pages import allPages
import numpy as np
from lxml import etree
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.preprocessing import scale
import math

if __name__=='__main__':
	UP_pages = allPages(["../Crawler/crawl_data/Questions/"])
	N_pages = len(UP_pages.pages)
	print "N_pages is " + str(N_pages)
	feature_matrix = []
	page = UP_pages.pages[101]
	#print page.tfidf
	all_text = ""
	tree = etree.HTML(str(page.original))
	#print etree.tostring(tree, pretty_print=True)
	for xpath_ in page.tfidf:
		if UP_pages.nidf[xpath_] <= 0.9 * N_pages and page.xpaths[xpath_]!=0:
			#idf = float(page.tfidf[xpath_])/float(page.xpaths[xpath_])
			#percentage = 1
			#threshold = math.log((float(1))/percentage,2)
			# delete nodes that appears less than percentage% of pages
			#if idf > threshold:
			print xpath_
			for non_stop in tree.xpath(xpath_):
				non_stop.getparent().remove(non_stop)
				#print stop
	#print all_text
	print etree.tostring(tree)
	clean_file = open("./output/clean_user.html","w")
	clean_file.write(etree.tostring(tree).replace("&#13;","\n"))
