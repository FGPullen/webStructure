import urllib2
import os
import traceback
import time
import random
from lxml import etree
#coding=utf8
header = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:32.0) Gecko/20100101 Firefox/32.0'}
#headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'} 

def intraJudge(url):
	# oulink with http or symbol like # and /
	# medhelp start from http://www.medhelp.org/user_groups/list and prefix http://www.medhelp.org/
	if len(url)==1 or "http" in url:
		if "medhelp.org" in url:
			return 1
		else:
			return 0
	elif url[0:2]=="//":
		return 0
	else:
		return 1
	'''
	if len(url)==1 or "http" in url:
		if "rottentomatoes.com" in url:
			return 1
		else:
			return 0
	elif url[0:2]=="//":
		return 0
	else:
		return 1

	if "http" in url:
		if "movie.douban.com" in url:
			return 1
		else:
			return 0
	else:
		if "ticket" in url:
			return 0
		else:
			return 1
	'''

def transform(url,prefix):
	if prefix not in url:
		return prefix + url
	else:
		return url 

def crawlUrl(url,url_stack,history_stack,prefix):
	if url in history_stack:
		print "Already crawled!"
		return 0
	else:
		#history_stack.append(url)
		response = urllib2.urlopen(url,timeout=30)
		lines = response.read().replace("\n","")
		#write to file
		#write_file = open("../crawl_data/Zhihu/"+url.replace("/","_")+".html",'w')
		file_name = "../crawl_data/medhelp2/"+url.replace("/","_")+".html"
		if os.path.isfile(file_name):
			print "Already"
			return 0
		write_file = open(file_name,'w')
		write_file.write(lines)

		# get links
		add_num = 0
		tree = etree.HTML(str(lines))
		Etree = etree.ElementTree(tree)
		nodes = tree.xpath("//a")
		for node in nodes:
			try:
				url_child = node.attrib['href']
				# if not intraJudge is for outlinks
				if intraJudge(url_child):
					url_child = transform(url_child,prefix)
					if url_child not in history_stack:
						url_stack.append(url_child)
					add_num += 1
			except:
				print "error"
				continue
		print "Succesfully crawled. %d urls are added." %add_num
		return 1

def stack2file(url_stack,history_stack,count):
	
	#stack_file = open("../stack_data/Zhihu/"+str(count/50)+".txt","w")
	stack_file = open("../stack_data/medhelp2/"+str(count/100)+".txt","w")
	stack_file.write("======url_stak======\n")
	for url in url_stack:
		stack_file.write(url+"\n")

	stack_file.write("======history_stak======\n")
	for url in history_stack:
		stack_file.write(url+"\n")	
	print "====save tempory stack status!===="

def getHistory(path,exception_html):
	history_stack = []
	lines = open(path,"r").readlines()
	for i in range(1,len(lines)):
		#print lines[i].replace("\n","")
		url = lines[i].replace("\n","")
		if url != exception_html:
			history_stack.append(url)
	return history_stack

error_log = open('error.txt','w')

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('inital_page', help='The entry page')
	parser.add_argument('prefix', help='For urls only have partial path')
	args = parser.parse_args()

	#url_stack = ["http://android.stackexchange.com/questions"]
	url_stack = [args.inital_page]
	#history_stack = getHistory("../stack_data/user_stack.txt",url_stack[0])
	history_stack = []
	#print len(history_stack)
	#prefix = "http://android.stackexchange.com"
	prefix = args.prefix
	print prefix +" is the prefix"
	count = 0
	count_save = 0


	while len(url_stack)!=0:
		try:
			first_url = url_stack[0]
			flag = crawlUrl(first_url,url_stack,history_stack,prefix)
			print "flag is %d" %flag
			print "length of url_stack is %d" %(len(url_stack))
			if flag:
				random_time_s = random.randint(5,15)
				time.sleep(random_time_s)
				count +=1
				count_save += 1

				if count%10==9:
					count = 0
					random_time = random.randint(150,210)
					print "---Already crawled %d pages! Take a rest for %d seconds---" %(len(history_stack),random_time)
					print "----We still have %d pages to crawl!----" %(len(url_stack))
					time.sleep(random_time)

				#if count_save%100==99:
				#	stack2file(url_stack,history_stack,count_save)

		except:
			print "error"
			traceback.print_exc()
			#error_log.write(url_stack[0] + " has a problem. We will skip this\n")
		finally:
			print "Pop and append"
			print first_url
			if first_url not in history_stack:
				history_stack.append(first_url)
			url_stack.pop(0)










