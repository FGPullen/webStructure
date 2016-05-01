# the aim of this program is sampling data
# start from a entry point and we follow links with some rules to get a better subset to construct sitemap
from lxml import etree
import re
import traceback
import random
import os
import cPickle as pickle
import urllib2
import time
from bs4 import BeautifulSoup


class crawler():

    def __init__(self,dataset,entry,prefix):
        self.history_set = set()
        self.entry = entry
        self.dataset = dataset
        self.prefix = prefix
        self.transition_dict = {}
        self.folder = "../../Crawler/full_data/" + dataset
        self.final_list = []
        self.num_page = 0
        self.page_num = {}



    def crawling(self):
        num_web_crawl = 0
        entry = self.entry
        self.url_stack  = [entry]
        num = 0
        crawl_id,flag = 0,0
        while(len(self.url_stack)>0):
            first_url = self.url_stack[0]
            #print "first url is " + first_url
            if first_url not in self.history_set:
                num += 1
                try:
                    self.crawl_link(first_url, self.url_stack, self.history_set)
                    self.final_list.append(first_url)
                    flag = 1
                except:
                    #print "crawl this page!"
                    #traceback.print_exc()
                    flag = self.crawlUrl(first_url,self.dataset,self.url_stack,self.history_set)
                    if flag == 1:
                        self.crawl_link(first_url, self.url_stack, self.history_set)
                        random_time_s = random.randint(5, 15)
                        num_web_crawl += 1
                        print "we have {0} page in total and {1} additionally crawled pages".format(num,num_web_crawl)
                        time.sleep(random_time_s)
                        if num_web_crawl%10 == 9:
                            #print "we have {0} page in total and {1} additionally crawled pages".format(num,num_web_crawl)
                            random_time_s = random.randint(60, 90)
                            time.sleep(random_time_s)
                    else:
                        num -= 1
                        print "crawling failed, pop and crawl later!"
            else:
                flag = 1

            self.url_stack.pop(0)
            if flag:
                self.history_set.add(first_url)
            else:
                self.url_stack.append(first_url)


    def crawl_link(self, first_url,url_stack, history_stack):
        self.num_page += 1
        file_path = self.folder + "/" + first_url.replace("/","_") +".html"
        #print file_path
        original = open(file_path,"r").read()
        contents = original.replace("\n","")
        link_dict = self.getAnchor(contents,first_url)
        #self.transition_dict[url] = link_dict
        for xpath in link_dict:
            link_list = link_dict[xpath]
            for url in link_list:
                #url = self.transform(url)
                #print url
                if url not in history_stack and url not in url_stack:
                    url_stack.append(url)
                    self.page_num[url] = self.num_page

    def transform(self,url):
        intra = self.intraJudge(url,self.dataset)
        if intra == 1:
            return self.prefix + url
        elif intra == 2:
            return url
        else:
            return url

    def getAnchor(self,contents,first_url):
        link_dict = {}
        tree= etree.HTML(str(contents))
        Etree = etree.ElementTree(tree)
        nodes = tree.xpath("//a")
        for node in nodes:
            try:
                xpath = self.removeIndex(Etree.getpath(node))
                #print xpath,node.attrib['href']
                if xpath not in link_dict:
                    link_dict[xpath] = []
                url = node.attrib['href']
                if self.intraJudge(url,self.dataset):
                    link_dict[xpath].append(self.transform(url))

            except:
                err = "Oh no! " + str(node)
        #print len(link_dict)
        new_link_dict = self.getlinks(link_dict)

        return new_link_dict

    def getlinks(self,link_dict):
        # this is a better link_dict which only contain intralink and contrain #samples (not now)
        new_link_dict = {}
        #print len(link_dict)
        for key in link_dict:
            links = link_dict[key]
            #print len(links)
            inlinks = []
            for link in links:
                inlinks.append(self.transform(link))

            ## inlink might be too many links, we have to sample at most four
            l = len(inlinks)

            if inlinks !=[]:
                #print len(inlinks)
                new_link_dict[key] = inlinks

        return new_link_dict


    def removeIndex(self,xpath):
        indexes = re.findall(r"\[\d+\]",str(xpath))
        for index in indexes:
            xpath = xpath.replace(index,"")
        return xpath

    def analyze_xpaths_dict(self):
        # input self.trans_dict and self.final_list
        total = 0
        exist = 0
        all_coverage_page = 0
        for key in self.transition_dict:
            flag = True
            xpath_dict = self.transition_dict[key]
            for xpath in xpath_dict:
                url_list = xpath_dict[xpath]
                total +=  len(url_list)
                for url in url_list:
                    if url in self.final_list:
                        exist += 1
                    else:
                        flag = False
            if flag:
                all_coverage_page += 1
        print str(float(exist)/float(total)) + " for coverage of links"
        print all_coverage_page
        print str(float(all_coverage_page)/float(self.size)) + "  the ratio of all covered page"

    def crawlUrl(self, url, site, url_stack, history_stack):
        if url in history_stack:
            print "Already crawled!"
            return 0
        else:
            try:
                response = urllib2.urlopen(url, timeout=30)
                lines = response.read().replace("\n", "")
                folder_path = "/bos/usr0/keyangx/webStructure/Crawler/full_data/" + site + "/"
                file_name = folder_path + url.replace("/", "_") + ".html"
            except:
                return 0
        #if ".html.html" in file_name:
        #    file_name = file_name.replace(".html.html", ".html")
        print file_name
        if os.path.isfile(file_name):
            print "Already"
            return 0
        try:
            write_file = open(file_name, 'w')
            write_file.write(lines)
        except:
            return 0
        print "succesfully crawled missing page!"
        print url

        return 1

    def intraJudge(self,url, site):
    # oulink with http or symbol like # and /
        if site == "stackexchange":
            if url[0]=="/" and url[0:2] !="//":
                return 1
            else:
                if "http://android.stackexchange.com/" in url:
                    return 2
                else:
                    return 0
        elif site == "yelp":
            if len(url) == 1 or "http" in url:
                if "http://www.yelp.com" in url:
                    return 0
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                print url
                return 1
        elif site == "asp":
            if len(url) == 1 or "http" in url:
                if "http://forums.asp.net" in url:
                    return 0
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                if url[0] == "/":
                    return 1
                else:
                    return 0
        elif site == "douban":
            if "http" in url:
                if "movie.douban.com" in url:
                    return 2
                else:
                    return 0
            else:
                return 0
        elif site == "tripadvisor":
            if "http" in url:
                if "tripadvisor.com" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                return 1
        elif site == "hupu":
            if "http" in url:
                if "voice.hupu.com" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                return 1
        elif site == "biketo":
            if "http" in url:
                if "biketo.com" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//" and not url.endswith(".jpg"):
                return 1
            else:
                return 0
        elif site == "amazon":
            if "http" in url:
                return 0
            elif url[0:2] == "//":
                return 0
            else:
                return 1
        elif site == "youtube":
            if "https://www.youtube.com" in url:
                return 0
            elif url[0:2] == "//":
                return 0
            else:
                if url[0:1] == "/":
                    return 1
                else:
                    return 0
        elif site == "csdn":  # http://bbs.csdn.net/home # http://bbs.csdn.net
            if "http" in url:
                if "my.csdn.net" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                if "javascript:void(0)" in url:
                    return 0
                else:
                    return 1
        elif site == "baidu":
            if "http" in url:
                if "tieba.baidu.com" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                if "javascript:void(0)" in url:
                    return 0
                else:
                    return 1
        elif site == "huffingtonpost":
            if "http" in url:
                if "http://www.huffingtonpost.com/" in url and not url.endswith(".jpg"):
                    return 2
                else:
                    return 0
        else:
            return 0

# must run on server for full data
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #"http://android.stackexchange.com/questions"
    parser.add_argument("dataset",choices=["stackexchange","rottentomatoes","youtube","asp","douban","tripadvisor","baidu","biketo","huffingtonpost","amazon"],help="the dataset to sample data from")
    parser.add_argument('entry', help='The entry page')
    parser.add_argument('prefix', help='For urls only have partial path')
    args = parser.parse_args()
    s = crawler(args.dataset,args.entry,args.prefix)
    s.crawling()
    #s.analyze_xpaths_dict()


    #print s.transition_dict
    #print s.final_list

