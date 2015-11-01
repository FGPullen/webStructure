import urllib2
import time
import random
from lxml import etree

#coding=utf8
header = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:32.0) Gecko/20100101 Firefox/32.0'}

def extractUser(uid):
    url = 'http://android.stackexchange.com/users/'
    url += urllib2.quote(query)
    response = urllib2.urlopen(url,timeout=30)
    lines = response.read()
    query_path = "../toy_data/users/user"+query+".html"
    write_file = open(query_path,'w')
    write_file.write(lines)


def extractQuestion(qid):
	url = "http://android.stackexchange.com/questions/"
	url += urllib2.quote(query)
	response = urllib2.urlopen(url,timeout=30)
	lines = response.read()
	query_path = "../toy_data/questions/question"+query+".html"
	write_file = open(query_path,'w')
	write_file.write(lines)


def  crawlerUserPages(queries):
	count = 0
	for query in queries:
		query = query.replace("\n","").split("\t")[1]
		try:
			extractUser(query)
			random_time_s = random.randint(10,20)
			time.sleep(random_time_s)
			print query
			count +=1
			if count%10==9:
				count = 0
				random_time = random.randint(240,360)
				time.sleep(random_time)
		except:
			error_log.write(query+" user\n")	

error_log = open('error.txt','w')

query_file = open("error_1.txt","r")
'''
queries = query_file.readlines()
count = 0
for query in queries:
	query = query.replace("\n","").split("\t")[1]
	try:
		extractUser(query)
		random_time_s = random.randint(10,20)
		time.sleep(random_time_s)
		print query
		count +=1
		if count%10==9:
			count = 0
			random_time = random.randint(240,360)
			time.sleep(random_time)
	except:
		error_log.write(query+" user\n")

'''
#query_file = open("qid.txt","r")
queries = query_file.readlines()
count = 0
for query in queries:
	#query = query.replace("\n","").split("\t")[1]
	query = query.replace("\n","").split(" ")[0]
	try:
		extractQuestion(query)
		random_time_s = random.randint(15,20)
		time.sleep(random_time_s)
		print "question \t" + str(query)
		count +=1
		if count%10==9:
			count = 0
			random_time = random.randint(100,120)
			time.sleep(random_time)
	except:
		error_log.write(query+" question\n")