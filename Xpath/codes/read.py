#coding=utf8
import os
import urllib
import time
from readability.readability import Document
web_data_path = "/Users/admin/CMU/codes/Xpath/data/trieschnigg1/trieschnigg1"
start_time = time.time()
num = 0 
for folder in os.listdir(web_data_path):
    if "." in folder:
        continue
    num += 1
    file_path = web_data_path + "/" + folder + "/index.html"
    html = urllib.urlopen(file_path).read()
    readable_article = Document(html).summary().encode("utf-8")
    readable_title = Document(html).short_title().encode("utf-8")
    new_page = open("../new_pages/" + folder + ".html","w")
    #new_page.write(readable_title + readable_article)
    #print file_path

time_interval = time.time() - start_time
print("--- %s seconds ---" % (time_interval))
mean_time = str(float(time_interval)/float(num))
print("--- Mean time %s seconds--" %(mean_time))
