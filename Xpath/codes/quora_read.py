#coding=utf8
import os
import urllib
import time
from readability.readability import Document
web_data_path = "/Users/admin/CMU/codes/Xpath/data/Quora"
start_time = time.time()
file_path = web_data_path + "/home.html"
html = urllib.urlopen(file_path).read()
readable_article = Document(html).summary().encode("utf-8")
readable_title = Document(html).short_title().encode("utf-8")
new_page = open("quora_home.html","w")
new_page.write(readable_title + readable_article)
    #print file_path
time_interval = time.time() - start_time
print("--- %s seconds ---" % (time_interval))
