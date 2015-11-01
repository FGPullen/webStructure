from page import Page
from pages import allPages
import re
# This program aims to calculate the possibility of occurence of two xpaths. They are also regarded as non-consecutive bigram
def removeIndex(xpath):
	indexes = re.findall(r"\[\d+\]",str(xpath))
	for index in indexes:
		xpath = xpath.replace(index,"")
	return xpath

if __name__=='__main__':
	#x1 = "/html/body/div/div/div/div/table/tr/td/div/div/div/ul/li/em"
	#x2 = "/html/body/div/div/div/div/table/tr/td/div/div/div/div/p/em/span"
	UP_pages = allPages(["../Crawler/crawl_data/showXpath/"])

	for page in UP_pages.pages:
		write_file = open("xpaths_demo/"+str(page.path).replace("../Crawler/crawl_data/showXpath/","")+".txt","w")
		xpath_list = page.dfs_xpaths_list
		last_path = ""
		count = 0
		for xpath in xpath_list:
			xpath = removeIndex(xpath)
			if xpath == last_path:
				count += 1
			else:
				if last_path !="":
					write_file.write(last_path + "\t" + str(count)+"\n")
				last_path = xpath
				count = 1
		#for xpath in xpath_list:
		#	print xpath

# /html/body/div/div/div/div/div/div/div/h3/span
# /html/body/div/div/div/div/div/div/div/h3
# /html/body/div/div/div/div/div/div/div/div/div/a/div/img
