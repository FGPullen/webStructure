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


class sampler():

    def __init__(self,dataset,entry,prefix,size,sample_num=1):
        self.history_set = set()
        self.size = size
        self.entry = entry
        self.dataset = dataset
        self.prefix = prefix
        self.transition_dict = {}
        self.folder = "../../Crawler/full_data/" + dataset
        self.final_list = []
        self.num_page = 0
        self.page_num = {}
        self.sample_num = sample_num
        self.html_stack = []
        self.url_stack = []
        '''
        test_file = self.folder + "/" + entry.replace("/","_")
        original = open(test_file,"r").read()
        contents = original.replace("\n","")
        link_dict = self.getAnchor(contents)
        for key in link_dict:
            print key, link_dict[key]
        '''
    #
    def crawling(self):
        num_web_crawl = 0
        entry = self.entry
        size = self.size
        self.url_stack  = [(entry,"","","")]
        self.html_stack = [entry]

        num = 0
        crawl_id = 0

        while(num<size and len(self.url_stack)>0):
            print num, size, "ohhoohho"
            first_url = self.url_stack[0][0]
            parent_url = self.url_stack[0][1]
            parent_xpath = self.url_stack[0][2]
            parent_index_xpath = self.url_stack[0][3]
            try:
                print "first url is " + first_url
            except:
                traceback.print_exc()
            if first_url not in self.history_set:
                num += 1
                try:
                    self.crawl_link(first_url,self.history_set)
                    self.final_list.append((first_url,parent_url,parent_xpath,parent_index_xpath))
                except:
                    print "might miss somthing here"
                    traceback.print_exc()
                    flag = self.crawlUrl(first_url,self.dataset,self.url_stack,self.history_set)
                    if flag == 1:
                        self.crawl_link(first_url,self.history_set)
                        self.final_list.append((first_url,parent_url,parent_xpath,parent_index_xpath))
                        random_time_s = random.randint(5, 15)
                        if self.dataset == "douban":
                            time.sleep(random_time_s*2)
                        time.sleep(random_time_s)
                        num_web_crawl += 1
                        if num_web_crawl%10 == 9:
                            random_time_s = random.randint(60, 120)
                            if self.dataset == "douban":
                                time.sleep(random_time_s)
                            time.sleep(random_time_s)
                    else:
                        num -= 1
            self.url_stack.pop(0)
            self.html_stack.pop(0)
            crawl_id += 1
            if num >= size:
                print "crawl_id is {0} for size {1}".format(crawl_id,size)
                print "first url comes from the {} th crawled page".format(self.page_num[first_url])
            print len(self.url_stack), "length of url_stack"
            self.history_set.add(first_url)

        if not os.path.exists('./trans_dict/'.format(self.dataset)):
            os.makedirs('./trans_dict/'.format(self.dataset))
        print './trans_dict/{}_trans.dict'.format(self.dataset)
        with open('./trans_dict/{}_trans.dict'.format(self.dataset), 'w') as outfile:
            pickle.dump(self.transition_dict, outfile, pickle.HIGHEST_PROTOCOL)

    def crawl_link(self, first_url, history_stack):
        self.num_page += 1
        if ".html" not in first_url:
            file_path = self.folder + "/" + first_url.replace("/","_") +".html"
        else:
            file_path = self.folder + "/" + first_url.replace("/","_")
        print file_path
        original = open(file_path,"r").read()
        contents = original.replace("\n","")
        link_dict = self.getAnchor(contents,first_url)
        #self.transition_dict[url] = link_dict
        for xpath in link_dict:
            link_tuple_list = link_dict[xpath]
            for tuple in link_tuple_list:
                url = tuple[0]
                index_xpath = tuple[1]
                #url = self.transform(url)
                #print url
                if url not in history_stack and url not in self.html_stack:
                    self.url_stack.append((url,first_url,xpath,index_xpath))
                    self.html_stack.append(url)
                    self.page_num[url] = self.num_page

    def transform(self,url):
        intra = self.intraJudge(url,self.dataset)
        if intra == 1:
            return self.prefix + url
        elif intra == 2:
            return url
        else:
            return url



    def getAnchor(self,contents,first_url,sample_flag=True):
        link_dict = {}
        tree= etree.HTML(str(contents))
        Etree = etree.ElementTree(tree)
        nodes = tree.xpath("//a")
        #print len(nodes), " number of nodes"
        for node in nodes:
            if 'class' in node.attrib:
                attrib = node.attrib['class']
            elif 'id' in node.attrib:
                attrib = node.attrib['id']
            else:
                attrib = ""
            try:
                index_xpath = Etree.getpath(node)
                xpath = self.removeIndex(index_xpath)
                xpath += "[{}]".format(attrib)
                #print xpath,node.attrib['href']
                if xpath not in link_dict:
                    link_dict[xpath] = []
                if 'href' in node.attrib:
                    url = node.attrib['href']
                else:
                    continue
                if self.intraJudge(url,self.dataset):
                    # for inital sampling
                    #link_dict[xpath].append((self.transform(url),first_url,xpath,index_xpath))
                    # for crawling
                    link_dict[xpath].append(self.transform(url))
            except:
                pass
        #print len(link_dict)
        print len(link_dict.keys()),"nodes number after filtering "
        self.transition_dict[first_url] = link_dict
        #print first_url, "hahahha", link_dict

        #print "!!! " + str(len(self.transition_dict)) + " " + first_url

        new_link_dict = self.getlinks(link_dict,sample_flag)
        #print "examine sampled link dict ", link_dict
        return new_link_dict


    def getlinks(self,link_dict,sample_flag):
        # this is a better link_dict which only contain intralink and contrain #samples (not now)
        new_link_dict = {}
        #print len(link_dict)
        for key in link_dict:
            links = link_dict[key]
            #print len(links)
            #print links
            '''
            inlinks = []
            for link in links:
                print link, "link triple"
                #raise
                inlinks.append(self.transform(link[0]))
            '''
            inlinks = links # only for convenience

            ## inlink might be too many links, we have to sample at most four
            if sample_flag:
                l = len(inlinks)
                if l > 1:
                    sub_links = []
                    inlinks = [ inlinks[i] for i in random.sample(xrange(len(inlinks)), l) ]
                    i = 0
                    while(len(sub_links)< self.sample_num and i < l): # only sample one from each xpath with attribute
                        link_triple = inlinks[i]
                        link = link_triple[0] # link, xpath, parent_url
                        file_path = self.folder + "/" + link.replace("/","_") +".html"

                        if os.path.isfile(file_path):
                            sub_links.append(link_triple)
                        i += 1
                    inlinks = sub_links

            if inlinks !=[]:
                #print len(inlinks)
                #print inlinks,"sample one"
                new_link_dict[key] = inlinks

        #print len(new_link_dict.keys())," number of new links"
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
                #folder_path = "/bos/usr0/keyangx/webStructure/Crawler/full_data/" + site + "/"
                folder_path = "../../Crawler/full_data/" + site + "/"
                file_name = folder_path + url.replace("/", "_") + ".html"
            except:
                traceback.print_exc()
                print " error in crawlUrl"
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
            if len(url) <= 1 or "http" in url:
                if "http://forums.asp.net" in url:
                    return 2
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
                if "javascript:void(0)" in url:
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
        elif site == "photozo":
            if "http" in url:
                if url.startswith("http://www.photozo.com") and not url.endswith(".jpg"):
                    return 2
            else:
                return 0
        elif site == "rottentomatoes":
            if "http" in url:
                if url.startswith("https://www.rottentomatoes.com"):
                    return 2
            elif url[0:2] == "//":
                return 0
            else:
                if url[0:1] == "/":
                    return 1
                else:
                    return 0
        else:
            return 0

# must run on server for full data
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #"http://android.stackexchange.com/questions"
    parser.add_argument("dataset",help="the dataset to sample data from")
    parser.add_argument('entry', help='The entry page')
    parser.add_argument('prefix', help='For urls only have partial path')
    args = parser.parse_args()
    s = sampler(args.dataset,args.entry,args.prefix,size=1000)
    s.crawling()
    folder_name = "./Jul30/site.sample"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    sample_file = open("{0}/{1}.sample".format(folder_name,s.dataset),"w")
    for link in s.final_list:
        url,parent_url,parent_xpath,index_xpath = link[0],link[1],link[2],link[3]
        sample_file.write(link[0] + "\t" + link[1] + "\t" + link[2]+ "\t" + link[3]  + "\n")
    #print len(s.transition_dict.keys())
    #s.analyze_xpaths_dict()


    #print s.transition_dict
    #print s.final_list

