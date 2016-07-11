import os
import sys
import collections
import lxml.html
import random
import matplotlib.pyplot as plt

def random_walk(site,site_dir,site_entry,prefix):

    url_queue = [site_entry]
    inlink_dict = collections.defaultdict(list)

    while url_queue!=[]:
        url = url_queue[0]
        inlink_set = []
        file_path = (site_dir + url.replace("/","_")) + ".html"
        #print file
        original = open(file_path,"r").read()
        contents = original.replace("\n","")
        try:
            root = lxml.html.fromstring(contents)
            hrefs = root.xpath('//a/@href')
            for href in hrefs:
                href_str = str(href).strip()
                if not href_str:
                    continue
                if href_str[0] == '#':
                    continue
                if href_str.startswith('javascript:'):
                    continue
                if href_str.startswith('mailto:'):
                    continue
                if href_str[0] == '/':
                    if len(href_str) > 1 and href_str[1] == '/':
                        continue
                    href_str = prefix + href_str

                if href.startswith('http'):
                    if not href.startswith(prefix):
                        continue
                #print "== ", href
                if not href.startswith('http'):
                    href = prefix + href
                inlink_set.append(href)
                print href
                inlink_dict[file] = inlink_set

            length = len(inlink_set)
            id = random.randrange(length)
            url = inlink_set[id]
            url_queue = [url]
            print id, url
        except:
            raise

if __name__ == '__main__':
    site = sys.argv[1]
    site_dir = sys.argv[2]
    site_entry = sys.argv[3]
    site_prefix = sys.argv[4]
    random_walk(site,site_dir,site_entry,site_prefix)